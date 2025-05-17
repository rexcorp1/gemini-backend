import { type ConnInfo, serve } from "https://deno.land/std@0.224.0/http/server.ts";
import { type SafetySetting, type GenerationConfig } from "npm:@google/generative-ai";

interface TextPart {
    text: string;
}

interface InlineDataPart {
    inlineData: {
        mimeType: string;
        data: string;
    };
}

interface FileDataPart {
    fileData: {
        mimeType: string;
        fileUri: string;
    };
}

type Part = TextPart | InlineDataPart | FileDataPart;

interface ChatRequestBody {
    parts: Part[];
    history: { role: string; parts: Part[] }[];
    model: string;
    generationConfig?: GenerationConfig;
    safetySettings?: SafetySetting[];
}

interface GoogleApiResponseChunk {
    candidates?: {
      content?: {
        parts?: TextPart[];
      };
      finishReason?: string;
      safetyRatings?: { category: string; probability: string }[];
    }[];
    promptFeedback?: {
      blockReason?: string;
      safetyRatings?: { category: string; probability: string }[];
    };
}

async function* processGoogleStream(stream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
    const reader = stream.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";

    const extractTextFromDataLine = (line: string): string | null => {
        if (line.startsWith("data: ")) {
            const jsonString = line.substring(6).trim();
            if (jsonString === "") return null;

            try {
                const chunk = JSON.parse(jsonString) as GoogleApiResponseChunk;
                const text = chunk.candidates?.[0]?.content?.parts?.[0]?.text;

                const finishReason = chunk.candidates?.[0]?.finishReason;
                if (finishReason && finishReason !== 'STOP') {
                    console.warn(`[${new Date().toISOString()}] Google API Finish Reason during stream: ${finishReason}`, chunk.candidates?.[0]?.safetyRatings ?? 'No safety ratings');
                }
                if (chunk.promptFeedback?.blockReason) {
                    console.warn(`[${new Date().toISOString()}] Google API Prompt Blocked during stream: ${chunk.promptFeedback.blockReason}`, chunk.promptFeedback.safetyRatings ?? 'No safety ratings');
                }
                return text ?? null;
            } catch (e) {
                console.error(`[${new Date().toISOString()}] Failed to parse JSON chunk:`, jsonString, e);
                return null;
            }
        } else if (line.trim() !== "") {
            console.warn(`[${new Date().toISOString()}] Received non-data line in SSE stream:`, line);
        }
        return null;
    };

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (value) {
                buffer += value;
                let newlineIndex;
                while ((newlineIndex = buffer.indexOf('\n')) >= 0) {
                    const singleLine = buffer.substring(0, newlineIndex);
                    buffer = buffer.substring(newlineIndex + 1);
                    const textChunk = extractTextFromDataLine(singleLine.trim());
                    if (textChunk !== null && textChunk !== undefined) {
                        yield textChunk;
                    }
                }
            }
            if (done) {
                if (buffer.trim()) {
                    const textChunk = extractTextFromDataLine(buffer.trim());
                    if (textChunk !== null && textChunk !== undefined) {
                        yield textChunk;
                    } else {
                         console.warn(`[${new Date().toISOString()}] Streaming finished with unprocessed buffer remnant (not valid data):`, buffer.trim());
                    }
                }
                break;
            }
        }
    } catch (error) {
        console.error(`[${new Date().toISOString()}] Error reading or processing Google API stream:`, error);
        throw error;
    } finally {
        reader.releaseLock();
    }
}

function isValidPart(part: unknown): part is Part {
    if (typeof part !== 'object' || part === null) return false;
    const hasText = 'text' in part && typeof (part as TextPart).text === 'string';
    const hasInlineData = 'inlineData' in part &&
                          typeof (part as InlineDataPart).inlineData === 'object' &&
                          (part as InlineDataPart).inlineData !== null &&
                          typeof (part as InlineDataPart).inlineData.mimeType === 'string' &&
                          typeof (part as InlineDataPart).inlineData.data === 'string';
    const hasFileData = 'fileData' in part &&
                        typeof (part as FileDataPart).fileData === 'object' &&
                        (part as FileDataPart).fileData !== null &&
                        typeof (part as FileDataPart).fileData.mimeType === 'string' &&
                        typeof (part as FileDataPart).fileData.fileUri === 'string';
    return hasText || hasInlineData || hasFileData;
}

function customEncodeToString(src: Uint8Array | string): string {
  if (typeof src === 'string') {
    return btoa(src);
  }
  const byteArray = new Uint8Array(src);
  let bin = '';
  byteArray.forEach((byte) => {
    bin += String.fromCharCode(byte);
  });
  return btoa(bin);
}

async function handler(req: Request, _connInfo: ConnInfo): Promise<Response> {
    const requestStartTime = Date.now();
    console.log(`[${new Date(requestStartTime).toISOString()}] Received request: ${req.method} ${req.url}`);

    const requestOrigin = req.headers.get("origin");
    let allowedOrigin = "null";

    const allowedOriginsList = [
        "http://localhost:8000",
        "https://gemini2-ashen.vercel.app"
    ];

    if (requestOrigin && allowedOriginsList.includes(requestOrigin)) {
        allowedOrigin = requestOrigin;
    }

    const corsHeaders = {
        'Access-Control-Allow-Origin': allowedOrigin,
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    };

    if (req.method === 'OPTIONS') {
        return new Response(null, { status: 204, headers: corsHeaders });
    }

    if (req.method !== 'POST') {
        return new Response(JSON.stringify({ error: `Method ${req.method} Not Allowed` }), {
            status: 405,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        });
    }

    const apiKey = Deno.env.get("GEMINI_API_KEY");
    if (!apiKey) {
        console.error("FATAL: GEMINI_API_KEY environment variable not set.");
        return new Response(JSON.stringify({ error: "Server configuration error: API key missing." }), {
            status: 500,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        });
    }

    if (req.headers.get("content-type")?.split(';')[0] !== 'application/json') {
        return new Response(JSON.stringify({ error: "Invalid Content-Type. Expected application/json" }), {
            status: 415,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        });
    }

    let body: ChatRequestBody;
    try {
        body = await req.json();
    } catch (e) {
        console.error(`[${new Date().toISOString()}] Failed to parse request body:`, e);
        return new Response(JSON.stringify({ error: "Invalid JSON in request body." }), {
            status: 400,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        });
    }

    const { parts: userParts, history, model, generationConfig, safetySettings } = body;

    const processedUserParts: Part[] = [];
    const imageRefRegex = /\[File Ref: (https?:\/\/[^\]]+)\]/g;

    for (const part of userParts) {
        if (part.text) {
            let lastIndex = 0;
            let match;
            const textSegments: string[] = [];
            let originalTextForPart = part.text;

            while ((match = imageRefRegex.exec(originalTextForPart)) !== null) {
                if (match.index > lastIndex) {
                    textSegments.push(originalTextForPart.substring(lastIndex, match.index));
                }
                const imageUrl = match[1];
                lastIndex = imageRefRegex.lastIndex;

                try {
                    console.log(`[${new Date().toISOString()}] Fetching image from URL: ${imageUrl}`);
                    const imageResponse = await fetch(imageUrl);

                    if (!imageResponse.ok) {
                        console.warn(`[${new Date().toISOString()}] Failed to fetch image ${imageUrl}: ${imageResponse.status}`);
                        textSegments.push(` [Failed to load image: ${imageResponse.statusText}] `);
                        continue;
                    }

                    const contentType = imageResponse.headers.get("content-type");
                    if (!contentType || !contentType.startsWith("image/")) {
                        console.warn(`[${new Date().toISOString()}] Fetched content from ${imageUrl} is not an image: ${contentType}`);
                        textSegments.push(` [Invalid image content: ${contentType}] `);
                        continue;
                    }

                    const imageBuffer = await imageResponse.arrayBuffer(); // This is an ArrayBuffer
                    const base64Data = customEncodeToString(imageBuffer); // Use the updated function

                    if (textSegments.length > 0) {
                        const combinedTextSegment = textSegments.join("").trim();
                        if (combinedTextSegment) processedUserParts.push({ text: combinedTextSegment });
                        textSegments.length = 0;
                    }

                    processedUserParts.push({
                        inlineData: { mimeType: contentType, data: base64Data },
                    });
                } catch (fetchError) {
                    console.error(`[${new Date().toISOString()}] Error fetching or processing image ${imageUrl}:`, fetchError);
                    textSegments.push(` [Error processing image: ${fetchError.message}] `);
                }
            }
            if (lastIndex < originalTextForPart.length) {
                textSegments.push(originalTextForPart.substring(lastIndex));
            }
            if (textSegments.length > 0) {
                const finalTextSegment = textSegments.join("").trim();
                if (finalTextSegment) processedUserParts.push({ text: finalTextSegment });
            }
        } else {
            processedUserParts.push(part);
        }
    }

    if (!model || typeof model !== 'string') {
        return new Response(JSON.stringify({ error: "Missing or invalid required field: model (string)" }), { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } });
    }
    // Validate userParts *after* processing, as processedUserParts might be empty if only an image was sent and failed to process
    if (processedUserParts.length === 0 && (!userParts || userParts.length === 0 || !userParts.some(p => p.text && p.text.trim() !== ''))) {
        return new Response(JSON.stringify({ error: "Missing or invalid required field: parts (non-empty array after processing)" }), { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } });
    }
    // Original userParts validation (can be kept for sanity check or removed if processedUserParts validation is sufficient)
    if (!Array.isArray(userParts) || !userParts.every(isValidPart)) {
         console.error("Bad Request: Invalid structure in original user parts array", userParts);
         return new Response(JSON.stringify({ error: "Invalid structure in original parts array. Each part must contain valid 'text', 'inlineData', or 'fileData'." }), { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } });
    }

    if (!Array.isArray(history)) {
         return new Response(JSON.stringify({ error: "Missing or invalid required field: history (array)" }), { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } });
    }
    if (!history.every(item =>
        item && typeof item.role === 'string' && (item.role === 'user' || item.role === 'model') &&
        Array.isArray(item.parts) && item.parts.every(isValidPart)
    )) {
        console.error("Bad Request: Invalid structure in history array", history);
        return new Response(JSON.stringify({ error: "Invalid structure in history array. Each item must have role ('user' or 'model') and an array of valid parts." }), { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } });
    }
    if (generationConfig !== undefined && typeof generationConfig !== 'object') {
        return new Response(JSON.stringify({ error: "Invalid optional field: generationConfig must be an object if provided." }), { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } });
    }
    if (safetySettings !== undefined && !Array.isArray(safetySettings)) {
        return new Response(JSON.stringify({ error: "Invalid optional field: safetySettings must be an array if provided." }), { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } });
    }

    const googleApiEndpoint = `https://generativelanguage.googleapis.com/v1beta/models/${model}:streamGenerateContent?key=${apiKey}&alt=sse`;

    const payload = {
        contents: [
            ...history, // Consider processing history parts for images too if needed
            { role: "user", parts: processedUserParts.length > 0 ? processedUserParts : userParts } // Use processed parts
        ],
        ...(generationConfig && { generationConfig }),
        ...(safetySettings && { safetySettings }),
    };

    console.log(`[${new Date().toISOString()}] Original userParts received by proxy:`, JSON.stringify(userParts));
    console.log(`[${new Date().toISOString()}] Forwarding to Google API. Processed user parts:`, JSON.stringify(processedUserParts));

    try {
        const googleApiResponse = await fetch(googleApiEndpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: AbortSignal.timeout(90000)
        });

        if (!googleApiResponse.ok || !googleApiResponse.body) {
             let errorBodyText = await googleApiResponse.text();
             let errorJson = null;
             let errorMessage = googleApiResponse.statusText;
             try {
                 errorJson = JSON.parse(errorBodyText);
                 errorMessage = errorJson?.error?.message || errorMessage;
                 console.error(`[${new Date().toISOString()}] Google API Error (${googleApiResponse.status}):`, errorJson);
             } catch (e) {
                 console.error(`[${new Date().toISOString()}] Google API Error (${googleApiResponse.status}): ${errorBodyText || '<empty body>'}`);
                 if (errorBodyText) errorMessage = errorBodyText;
             }
            return new Response(JSON.stringify({ error: `Google API Error: ${errorMessage}` }), {
                status: googleApiResponse.status,
                headers: { ...corsHeaders, 'Content-Type': 'application/json' },
            });
        }

        console.log(`[${new Date().toISOString()}] Google API Streaming Success Response (${googleApiResponse.status}) received for model ${model}. Starting stream processing.`);

        const responseStream = new ReadableStream({
            async start(controller) {
                const streamStartTime = Date.now();
                try {
                    const encoder = new TextEncoder();
                    for await (const textChunk of processGoogleStream(googleApiResponse.body!)) {
                        controller.enqueue(encoder.encode(textChunk));
                    }
                    controller.close();
                    const streamEndTime = Date.now();
                    console.log(`[${new Date().toISOString()}] Deno response stream closed successfully for model ${model}. Processing time: ${streamEndTime - streamStartTime}ms`);
                } catch (streamError) {
                     const streamEndTime = Date.now();
                     console.error(`[${new Date().toISOString()}] Error processing Google stream or writing to response:`, streamError, `Stream active for ${streamEndTime - streamStartTime}ms`);
                     if (controller.desiredSize === null || (controller.desiredSize && controller.desiredSize > 0)) {
                         try {
                             const errorMsg = `\n[STREAM_ERROR] Processing failed: ${streamError.message}\n`;
                             controller.enqueue(new TextEncoder().encode(errorMsg));
                         } catch (enqueueError) {
                             console.error("Failed to enqueue stream error message:", enqueueError);
                         }
                     }
                     controller.error(streamError);
                }
            }
        });

        return new Response(responseStream, {
            status: 200,
            headers: {
                ...corsHeaders,
                'Content-Type': 'text/plain; charset=utf-8',
                'X-Content-Type-Options': 'nosniff',
                'Cache-Control': 'no-cache',
            }
        });

    } catch (error) {
        const requestEndTime = Date.now();
        console.error(`[${new Date().toISOString()}] Unhandled Error in Deno handler:`, error, `Total request time: ${requestEndTime - requestStartTime}ms`);
        let status = 500;
        let message = `Internal Server Error: ${error.message}`;

        if (error.name === 'TimeoutError') {
            status = 504;
            message = 'Request to Google API timed out.';
        } else if (error.message.startsWith('Stream stopped by API due to:') || error.message.startsWith('Request blocked by API due to:')) {
             status = 400;
             message = error.message;
         }

        return new Response(JSON.stringify({ error: message }), {
            status: status,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        });
    }
}

const port = 8000;
console.log(`Chat server starting on http://localhost:${port}`);
serve(handler, { port });
