// main.ts

import { type ConnInfo, serve } from "https://deno.land/std@0.224.0/http/server.ts"; // Gunakan versi std terbaru jika perlu
import { type SafetySetting, type GenerationConfig } from "npm:@google/generative-ai"; // Impor tipe

// --- Definisi Tipe untuk Multimodal ---
interface TextPart {
    text: string;
}

interface InlineDataPart {
    inlineData: {
        mimeType: string; // e.g., "image/png", "image/jpeg"
        data: string;     // Base64 encoded string
    };
}

interface FileDataPart {
    fileData: {
        mimeType: string; // e.g., "audio/wav", "video/mp4"
        fileUri: string;  // URI dari file yang diunggah (via File API atau GCS)
    };
}

// Tipe Part adalah gabungan dari kemungkinan tipe bagian dalam request
type Part = TextPart | InlineDataPart | FileDataPart;

// Definisikan tipe untuk request body agar lebih jelas
interface ChatRequestBody {
    parts: Part[]; // Menggunakan array 'parts' untuk input pengguna
    history: { role: string; parts: Part[] }[]; // History juga menggunakan Part[]
    model: string;
    generationConfig?: GenerationConfig; // Opsional
    safetySettings?: SafetySetting[];   // Opsional
}

// --- Tipe Internal untuk Google API Response (untuk parsing JSON) ---
// Ini membantu dalam type safety saat memproses stream
interface GoogleApiResponseChunk {
    candidates?: {
      content?: {
        parts?: TextPart[]; // Asumsi respons stream hanya teks untuk saat ini
      };
      finishReason?: string;
      safetyRatings?: { category: string; probability: string }[];
    }[];
    promptFeedback?: {
      blockReason?: string;
      safetyRatings?: { category: string; probability: string }[];
    };
  }

// Fungsi untuk memproses stream dari Google API
async function* processGoogleStream(stream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
    const reader = stream.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";

    const processLine = (line: string): string | null => { // Helper function to process a single line
        if (line.startsWith("data: ")) {
            const jsonString = line.substring(6).trim();
            if (jsonString === "") return null; // Skip empty data lines

            try {
                const chunk: GoogleApiResponseChunk = JSON.parse(jsonString);
                const text = chunk.candidates?.[0]?.content?.parts?.[0]?.text;

                // Check for finish/block reasons even if there's no text in this specific chunk
                const finishReason = chunk.candidates?.[0]?.finishReason;
                if (finishReason && finishReason !== 'STOP') {
                    console.warn(`[${new Date().toISOString()}] Google API Finish Reason during stream: ${finishReason}`, chunk.candidates?.[0]?.safetyRatings ?? 'No safety ratings');
                    // Optionally throw or handle differently
                }
                if (chunk.promptFeedback?.blockReason) {
                    console.warn(`[${new Date().toISOString()}] Google API Prompt Blocked during stream: ${chunk.promptFeedback.blockReason}`, chunk.promptFeedback.safetyRatings ?? 'No safety ratings');
                    // Optionally throw or handle differently
                }

                return text ?? null; // Return the text part, could be null/undefined
            } catch (e) {
                console.error(`[${new Date().toISOString()}] Failed to parse JSON chunk:`, jsonString, e);
                return null; // Skip this chunk on error
            }
        } else if (line.trim() !== "") {
            console.warn(`[${new Date().toISOString()}] Received non-data line in SSE stream:`, line);
        }
        return null; // Not a data line we process for text
    };

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (value) { // Process value if present
                buffer += value;
                let newlineIndex;
                // Process lines separated by '\n'
                // Note: Standard SSE uses '\n\n' to separate messages. This assumes each 'data:' line is a complete message or Google's stream format differs.
                while ((newlineIndex = buffer.indexOf('\n')) >= 0) {
                    const line = buffer.substring(0, newlineIndex).trim();
                    buffer = buffer.substring(newlineIndex + 1);
                    const textChunk = processLine(line);
                    if (textChunk) {
                        yield textChunk;
                    }
                }
            }
            if (done) {
                // Process any remaining part in the buffer after the stream is done
                if (buffer.trim()) {
                    const textChunk = processLine(buffer.trim());
                    if (textChunk) {
                        yield textChunk;
                    } else {
                         // Log if the remnant wasn't valid data we could parse text from
                         console.warn(`[${new Date().toISOString()}] Streaming finished with unprocessed buffer remnant (not valid data):`, buffer.trim());
                    }
                }
                break; // Exit the loop
            }
        }
    } catch (error) {
        console.error(`[${new Date().toISOString()}] Error reading or processing Google API stream:`, error);
        throw error; // Re-throw to be handled by the main handler
    } finally {
        reader.releaseLock(); // Pastikan reader dilepaskan
    }
}

// Fungsi pembantu untuk membuat respons error JSON
function createErrorResponse(message: string, status: number): Response {
    return new Response(JSON.stringify({ error: message }), {
        status: status,
        headers: { 'Content-Type': 'application/json' },
    });
}

// Fungsi pembantu untuk validasi struktur Part
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

// Handler utama untuk request HTTP
async function handler(req: Request, _connInfo: ConnInfo): Promise<Response> {
    const requestStartTime = Date.now();
    console.log(`[${new Date(requestStartTime).toISOString()}] Received request: ${req.method} ${req.url}`);

    if (req.method !== 'POST') {
        return createErrorResponse(`Method ${req.method} Not Allowed`, 405);
    }

    // Periksa Content-Type
    if (req.headers.get("content-type")?.split(';')[0] !== 'application/json') {
         return createErrorResponse("Invalid Content-Type. Expected application/json", 415);
    }

    const apiKey = Deno.env.get("GEMINI_API_KEY");
    if (!apiKey) {
        console.error("FATAL: GEMINI_API_KEY environment variable not set.");
        return createErrorResponse("Server configuration error: API key missing.", 500);
    }

    let body: ChatRequestBody;
    try {
        body = await req.json();
    } catch (e) {
        console.error(`[${new Date().toISOString()}] Failed to parse request body:`, e);
        return createErrorResponse("Invalid JSON in request body.", 400);
    }

    // Destructuring dengan nama yang lebih jelas
    const { parts: userParts, history, model, generationConfig, safetySettings } = body;

    // --- Validasi Input yang Lebih Rinci ---
    if (!model || typeof model !== 'string') {
        return createErrorResponse("Missing or invalid required field: model (string)", 400);
    }
    if (!Array.isArray(userParts) || userParts.length === 0) {
        return createErrorResponse("Missing or invalid required field: parts (non-empty array)", 400);
    }
    if (!userParts.every(isValidPart)) {
         console.error("Bad Request: Invalid structure in user parts array", userParts);
         return createErrorResponse("Invalid structure in parts array. Each part must contain valid 'text', 'inlineData', or 'fileData'.", 400);
    }
    if (!Array.isArray(history)) {
         return createErrorResponse("Missing or invalid required field: history (array)", 400);
    }
    // Validasi struktur history (lebih ketat)
    if (!history.every(item =>
        item && typeof item.role === 'string' && (item.role === 'user' || item.role === 'model') &&
        Array.isArray(item.parts) && item.parts.every(isValidPart)
    )) {
         console.error("Bad Request: Invalid structure in history array", history);
         return createErrorResponse("Invalid structure in history array. Each item must have role ('user' or 'model') and an array of valid parts.", 400);
    }
    // Validasi opsional (contoh: pastikan generationConfig adalah objek jika ada)
    if (generationConfig !== undefined && typeof generationConfig !== 'object') {
        return createErrorResponse("Invalid optional field: generationConfig must be an object if provided.", 400);
    }
    if (safetySettings !== undefined && !Array.isArray(safetySettings)) {
        return createErrorResponse("Invalid optional field: safetySettings must be an array if provided.", 400);
    }
    // TODO: Tambahkan validasi lebih dalam untuk safetySettings jika perlu


    // Gunakan endpoint streaming
    const googleApiEndpoint = `https://generativelanguage.googleapis.com/v1beta/models/${model}:streamGenerateContent?key=${apiKey}&alt=sse`;

    const payload = {
        contents: [
            ...history, // Sertakan history sebelumnya
            {
                role: "user",
                parts: userParts // Gunakan parts dari request saat ini
            }
        ],
        // Sertakan konfigurasi hanya jika ada di request
        ...(generationConfig && { generationConfig }),
        ...(safetySettings && { safetySettings }),
    };

    console.log(`[${new Date().toISOString()}] Forwarding to Google API (Streaming): ${googleApiEndpoint.split('?')[0]} for model ${model} with ${userParts.length} user parts.`);
    // console.debug("Payload:", JSON.stringify(payload)); // Hati-hati log data base64 yang besar

    try {
        const googleApiResponse = await fetch(googleApiEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                // Pertimbangkan menambahkan User-Agent kustom
                // 'User-Agent': 'MyDenoApp/1.0'
            },
            body: JSON.stringify(payload),
            signal: AbortSignal.timeout(90000) // Timeout lebih lama untuk streaming (90 detik)
        });

        if (!googleApiResponse.ok || !googleApiResponse.body) {
             // Coba baca body error jika ada untuk logging yang lebih baik
             let errorBodyText = await googleApiResponse.text(); // Baca sebagai teks dulu
             let errorJson = null;
             let errorMessage = googleApiResponse.statusText; // Default message
             try {
                 errorJson = JSON.parse(errorBodyText);
                 errorMessage = errorJson?.error?.message || errorMessage; // Coba ambil pesan dari JSON
                 console.error(`[${new Date().toISOString()}] Google API Error (${googleApiResponse.status}):`, errorJson);
             } catch (e) {
                 // Jika body bukan JSON atau kosong
                 console.error(`[${new Date().toISOString()}] Google API Error (${googleApiResponse.status}): ${errorBodyText || '<empty body>'}`);
                 if (errorBodyText) errorMessage = errorBodyText; // Gunakan teks jika bukan JSON
             }

            // Kembalikan error ke klien
            return createErrorResponse(`Google API Error: ${errorMessage}`, googleApiResponse.status); // Gunakan status code dari Google
        }

        console.log(`[${new Date().toISOString()}] Google API Streaming Success Response (${googleApiResponse.status}) received for model ${model}. Starting stream processing.`);

        // Buat ReadableStream untuk respons Deno
        const responseStream = new ReadableStream({
            async start(controller) {
                const streamStartTime = Date.now();
                try {
                    const encoder = new TextEncoder();
                    for await (const textChunk of processGoogleStream(googleApiResponse.body!)) {
                        // console.debug("Yielding chunk:", textChunk); // Untuk debugging stream
                        controller.enqueue(encoder.encode(textChunk));
                    }
                    controller.close();
                    const streamEndTime = Date.now();
                    console.log(`[${new Date().toISOString()}] Deno response stream closed successfully for model ${model}. Processing time: ${streamEndTime - streamStartTime}ms`);
                } catch (streamError) {
                     const streamEndTime = Date.now();
                     console.error(`[${new Date().toISOString()}] Error processing Google stream or writing to response:`, streamError, `Stream active for ${streamEndTime - streamStartTime}ms`);
                     // Coba kirim pesan error ke client jika stream masih terbuka dan belum error
                     if (!controller.desiredSize === null || controller.desiredSize > 0) {
                         try {
                             // Format error sebagai teks biasa atau JSON, tergantung preferensi klien
                             const errorMsg = `\n[STREAM_ERROR] Processing failed: ${streamError.message}\n`;
                             controller.enqueue(new TextEncoder().encode(errorMsg));
                         } catch (enqueueError) {
                             console.error("Failed to enqueue stream error message:", enqueueError);
                         }
                     }
                     // Tutup stream dengan error agar klien tahu ada masalah
                     controller.error(streamError);
                }
            }
        });

        // Kirim stream sebagai respons
        return new Response(responseStream, {
            status: 200, // Sukses memulai streaming
            headers: {
                'Content-Type': 'text/plain; charset=utf-8', // Kirim sebagai teks biasa, potongan demi potongan
                'X-Content-Type-Options': 'nosniff',
                'Cache-Control': 'no-cache', // Penting untuk streaming
                // Jika Anda ingin format SSE dari server Deno ke klien:
                // 'Content-Type': 'text/event-stream; charset=utf-8',
                // Dan ubah enqueue di atas menjadi format SSE:
                // controller.enqueue(encoder.encode(`data: ${JSON.stringify({text: textChunk})}\n\n`));
            }
        });

    } catch (error) {
        const requestEndTime = Date.now();
        console.error(`[${new Date().toISOString()}] Unhandled Error in Deno handler:`, error, `Total request time: ${requestEndTime - requestStartTime}ms`);
        let status = 500;
        let message = `Internal Server Error: ${error.message}`;

        if (error.name === 'TimeoutError') {
            status = 504; // Gateway Timeout
            message = 'Request to Google API timed out.';
        } else if (error.message.startsWith('Stream stopped by API due to:') || error.message.startsWith('Request blocked by API due to:')) {
             // Jika error dilempar dari processGoogleStream karena alasan API
             status = 400; // Atau 502 Bad Gateway, tergantung bagaimana Anda menginterpretasikannya
             message = error.message;
         }

        return createErrorResponse(message, status);
    }
}

// Jalankan server
const port = 8000;
console.log(`Chat server starting on http://localhost:${port}`);
serve(handler, { port });
