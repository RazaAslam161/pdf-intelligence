# Frontend

React + TypeScript + Vite UI for the PDF Q&A app. Plain CSS / CSS Modules, no UI framework.

## Setup

```bash
cd frontend
npm install
cp .env.example .env   # set VITE_API_BASE if the backend isn't on :8000
```

## Scripts

```bash
npm run dev        # dev server on http://localhost:5173
npm run build      # type-check + production build
npm run preview    # preview the production build
npm test           # unit tests (Vitest)
```

## Environment

| Variable | Default | Description |
| --- | --- | --- |
| `VITE_API_BASE` | `http://localhost:8000` | Backend base URL |
| `VITE_USE_SAMPLE` | `false` | Set `true` to render canned sample data without a backend |
