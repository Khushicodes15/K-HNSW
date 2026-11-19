/// <reference lib="webworker" />

import { parseEventTrace } from "../replay/trace";

interface ParseRequest {
  id: number;
  text: string;
}

self.addEventListener("message", (event: MessageEvent<ParseRequest>) => {
  try {
    const events = parseEventTrace(event.data.text);
    self.postMessage({ id: event.data.id, events });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Trace parsing failed.";
    self.postMessage({ id: event.data.id, error: message });
  }
});
