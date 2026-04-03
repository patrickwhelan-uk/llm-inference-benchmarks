# Hardware Profile: M2 Mac Studio

## GPU (Apple Silicon — Unified Memory Architecture)

| Field | Value |
|-------|-------|
| Chip | Apple M2 Max |
| GPU Cores | 30 |
| Neural Engine Cores | 16 |
| Metal Support | Metal 3 |
| Memory Bandwidth | 400 GB/s |

## CPU

| Field | Value |
|-------|-------|
| Chip | Apple M2 Max |
| Performance Cores | 8 |
| Efficiency Cores | 4 |
| Total Cores | 12 |

## Memory (Unified)

| Field | Value |
|-------|-------|
| Capacity | 64 GB |
| Type | LPDDR5 |
| Bandwidth | 400 GB/s |
| Configuration | Unified (shared CPU/GPU) |

## Storage

| Field | Value |
|-------|-------|
| Primary | 1 TB Apple SSD |

## Operating System

| Field | Value |
|-------|-------|
| OS | macOS |
| Version | Sequoia 15.x |

## Inference Stack

| Component | Value |
|-----------|-------|
| Server | ai-gateway (FastAPI) |
| Backend | vllm-mlx |
| Framework | MLX |
| Quantization | 4-bit (mlx-community models) |

## Notes

- Unified memory architecture: CPU and GPU share the same 64 GB memory pool
- No discrete GPU — all inference runs on the M2 Max integrated GPU
- Memory monitoring via `vm_stat` (system-wide, not per-process)
- Power monitoring not available without `sudo powermetrics`
