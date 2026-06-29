# PDF Intelligence — Frontend

React + TypeScript + Vite frontend for the RAG PDF chatbot. Hand-crafted UI in
the "Marginalia Console" design direction (dark-first research terminal). Plain
CSS + CSS Modules, no UI framework.

## Requirements

- Node 18+ (developed on Node 24)
- npm 9+

## Setup

```bash
cd frontend
npm install
cp .env.example .env   # then edit VITE_API_BASE if your backend is not on :8000
```

## Scripts

```bash
npm run dev        # start the Vite dev server (http://localhost:5173)
npm run build      # type-check (tsc --noEmit) then production build
npm run preview    # preview the production build locally
npm run typecheck  # type-check only
```

## Environment

| Variable        | Default                 | Description                       |
| --------------- | ----------------------- | --------------------------------- |
| `VITE_API_BASE` | `http://localhost:8000` | Base URL of the FastAPI backend.  |

## Status

This is the design-system foundation (milestone B3): design tokens, app shell,
base components, a typed API client stub, and an empty conversation state. Chat
and citation logic are intentionally not implemented yet.
