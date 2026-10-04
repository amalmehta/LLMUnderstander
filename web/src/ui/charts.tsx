import { JSX } from 'preact'
import { useWidth } from './kit'

// Small SVG charts drawn to scale with one linear scale per axis. Colours come from theme tokens.

export interface Series { name: string; color: string; points: { x: number; y: number }[]; width?: number; dashed?: boolean; dots?: boolean; opacity?: number }
export interface Rule { x?: number; y?: number; color?: string; label?: string; dashed?: boolean }

const niceTicks = (lo: number, hi: number, n = 5): number[] => {
  if (!(hi > lo)) return [lo]
  const raw = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw
  const out: number[] = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12))
  return out
}

const defaultFmt = (v: number) => {
  const a = Math.abs(v)
  if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 0 : 1) + 'M'
  if (a >= 1e3) return (v / 1e3).toFixed(a >= 1e4 ? 0 : 1) + 'K'
  if (a > 0 && a < 0.01) return v.toExponential(0)
  return String(+v.toPrecision(3))
}

export function LineChart({ series, height = 220, xDomain, yDomain, xLabel, yLabel, rules = [], xFormat = defaultFmt, yFormat = defaultFmt, xTicks, legend = true, onClick }: {
  series: Series[]; height?: number; xDomain?: [number, number]; yDomain?: [number, number]; xLabel?: string; yLabel?: string; rules?: Rule[];
  xFormat?: (v: number) => string; yFormat?: (v: number) => string; xTicks?: number[]; legend?: boolean; onClick?: (x: number) => void
}) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const all = series.flatMap((s) => s.points)
  const xd = xDomain ?? [Math.min(...all.map((p) => p.x), 0), Math.max(...all.map((p) => p.x), 1)]
  const yd = yDomain ?? [Math.min(...all.map((p) => p.y), 0), Math.max(...all.map((p) => p.y), 1)]
  const m = { l: 46, r: 12, t: 10, b: xLabel ? 36 : 22 }
  const W = width, H = height, iw = W - m.l - m.r, ih = H - m.t - m.b
  const sx = (x: number) => m.l + ((x - xd[0]) / (xd[1] - xd[0] || 1)) * iw
  const sy = (y: number) => m.t + ih - ((Math.min(Math.max(y, yd[0]), yd[1]) - yd[0]) / (yd[1] - yd[0] || 1)) * ih
  const xt = xTicks ?? niceTicks(xd[0], xd[1], Math.max(2, Math.floor(iw / 90)))
  const yt = niceTicks(yd[0], yd[1], 4)
  return (
    <div ref={ref}>
      <svg class="chart" width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={yLabel ? `${yLabel} chart` : 'chart'}
        onClick={onClick ? (e) => { const r = (e.currentTarget as SVGElement).getBoundingClientRect(); onClick(xd[0] + ((e.clientX - r.left - m.l) / iw) * (xd[1] - xd[0])) } : undefined}>
        {yt.map((v) => <g key={'y' + v}><line class="grid" x1={m.l} x2={W - m.r} y1={sy(v)} y2={sy(v)} /><text x={m.l - 6} y={sy(v) + 4} text-anchor="end">{yFormat(v)}</text></g>)}
        {xt.map((v) => <text key={'x' + v} x={sx(v)} y={H - m.b + 15} text-anchor="middle">{xFormat(v)}</text>)}
        {xLabel && <text x={m.l + iw / 2} y={H - 4} text-anchor="middle">{xLabel}</text>}
        {yLabel && <text x={m.l} y={m.t - 0} text-anchor="start" dy="-2">{yLabel}</text>}
        {rules.map((r, i) => r.y !== undefined
          ? <g key={'r' + i}><line x1={m.l} x2={W - m.r} y1={sy(r.y)} y2={sy(r.y)} stroke={r.color ?? 'var(--muted)'} stroke-dasharray={r.dashed === false ? undefined : '5 4'} />{r.label && <text x={m.l + 4} y={sy(r.y) - 4} style={{ fill: r.color }}>{r.label}</text>}</g>
          : <line key={'r' + i} x1={sx(r.x!)} x2={sx(r.x!)} y1={m.t} y2={m.t + ih} stroke={r.color ?? 'var(--muted)'} stroke-dasharray={r.dashed ? '5 4' : undefined} opacity="0.6" />)}
        {series.map((s) => s.points.length > 0 && (
          <g key={s.name}>
            <polyline fill="none" stroke={s.color} stroke-width={s.width ?? 2} stroke-dasharray={s.dashed ? '5 4' : undefined} opacity={s.opacity ?? 1} stroke-linejoin="round"
              points={s.points.map((p) => `${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ')} />
            {s.dots && s.points.map((p, i) => <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r="3" fill={s.color} />)}
          </g>
        ))}
      </svg>
      {legend && series.filter((s) => (s.opacity ?? 1) > 0.5).length > 1 && (
        <div class="legend">{series.filter((s) => (s.opacity ?? 1) > 0.5).map((s) => <span key={s.name}><i style={{ background: s.color }} />{s.name}</span>)}</div>
      )}
    </div>
  )
}

export interface Bar { label: string; value: number; color?: string; note?: string }

/** Vertical bars with value labels. */
export function BarChart({ bars, height = 200, yDomain, yFormat = defaultFmt, valueFormat }: { bars: Bar[]; height?: number; yDomain?: [number, number]; yFormat?: (v: number) => string; valueFormat?: (v: number) => string }) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const yd = yDomain ?? [Math.min(0, ...bars.map((b) => b.value)), Math.max(1e-9, ...bars.map((b) => b.value))]
  const m = { l: 40, r: 6, t: 16, b: 24 }, W = width, H = height, iw = W - m.l - m.r, ih = H - m.t - m.b
  const sy = (y: number) => m.t + ih - ((y - yd[0]) / (yd[1] - yd[0])) * ih
  const bw = iw / Math.max(bars.length, 1)
  return (
    <div ref={ref}>
      <svg class="chart" width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="bar chart">
        {niceTicks(yd[0], yd[1], 4).map((v) => <g key={v}><line class="grid" x1={m.l} x2={W - m.r} y1={sy(v)} y2={sy(v)} /><text x={m.l - 6} y={sy(v) + 4} text-anchor="end">{yFormat(v)}</text></g>)}
        {bars.map((b, i) => {
          const x = m.l + i * bw + bw * 0.14, w = bw * 0.72, y0 = sy(Math.max(0, yd[0])), y1 = sy(b.value)
          return (
            <g key={i}>
              <rect x={x} y={Math.min(y0, y1)} width={w} height={Math.max(1, Math.abs(y0 - y1))} rx="3" fill={b.color ?? 'var(--c1)'} />
              {valueFormat && <text x={x + w / 2} y={Math.min(y0, y1) - 4} text-anchor="middle" style={{ fontSize: '10px' }}>{valueFormat(b.value)}</text>}
              <text x={x + w / 2} y={H - 8} text-anchor="middle" style={{ fontFamily: 'var(--font-mono)', fontSize: '11.5px', fill: 'var(--ink)' }}>{b.label}</text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

/** Horizontal bars with a label above each bar (fits long labels on phones). */
export function HBarChart({ bars, domain, format, rowHeight = 34, markers }: { bars: Bar[]; domain?: [number, number]; format?: (v: number) => string; rowHeight?: number; markers?: number[] }) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const d = domain ?? [Math.min(0, ...bars.map((b) => b.value)), Math.max(1e-9, ...bars.map((b) => b.value))]
  const labelRoom = format ? 64 : 8
  const W = width, iw = W - labelRoom, H = bars.length * rowHeight + 4
  const sx = (v: number) => ((v - d[0]) / (d[1] - d[0])) * iw
  const zero = sx(Math.max(d[0], Math.min(0, d[1])))
  return (
    <div ref={ref}>
      <svg class="chart" width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="bar chart">
        {bars.map((b, i) => {
          const y = i * rowHeight, x1 = sx(b.value)
          return (
            <g key={i}>
              <text x="0" y={y + 12} style={{ fill: 'var(--ink)', fontSize: '12px' }}>{b.label}{b.note ? <tspan style={{ fill: 'var(--muted)' }}>{'  ' + b.note}</tspan> : null}</text>
              <rect x={Math.min(zero, x1)} y={y + 16} width={Math.max(1.5, Math.abs(x1 - zero))} height={rowHeight - 22} rx="3" fill={b.color ?? 'var(--c1)'} />
              {markers && markers[i] !== undefined && <path d={`M${sx(markers[i])} ${y + 14} l5 ${(rowHeight - 18) / 2} l-5 ${(rowHeight - 18) / 2} l-5 ${-(rowHeight - 18) / 2} Z`} fill="var(--muted)" />}
              {format && <text x={Math.max(zero, x1) + 6} y={y + rowHeight - 9} style={{ fontSize: '11px' }}>{format(b.value)}</text>}
            </g>
          )
        })}
      </svg>
    </div>
  )
}

/** A grid of coloured cells. Rows can be clicked. */
export function Heatmap({ values, rowLabels = [], colLabels = [], diverging = false, cell = 18, highlightRow, onRow, showNumbers = false }: {
  values: number[][]; rowLabels?: string[]; colLabels?: string[]; diverging?: boolean; cell?: number; highlightRow?: number; onRow?: (r: number) => void; showNumbers?: boolean
}) {
  const rows = values.length, cols = values[0]?.length ?? 0
  const scale = Math.max(1e-9, ...values.flat().map(Math.abs))
  const lw = rowLabels.length ? 20 : 0, lh = colLabels.length ? 18 : 0
  const W = lw + cols * cell, H = lh + rows * cell
  const color = (v: number): string => {
    const t = Math.max(-1, Math.min(1, v / scale))
    if (diverging) return `color-mix(in srgb, ${t >= 0 ? 'var(--pos)' : 'var(--neg)'} ${Math.round(Math.abs(t) * 85)}%, var(--zero))`
    return `color-mix(in srgb, var(--heat-hi) ${Math.round(Math.max(0, t) * 100)}%, var(--heat-lo))`
  }
  return (
    <div class="scroll-x">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: 'block', fontFamily: 'var(--font-mono)' }} role="img" aria-label="heatmap">
        {colLabels.map((l, c) => <text key={'c' + c} x={lw + (c + 0.5) * cell} y={12} text-anchor="middle" style={{ fontSize: '11px', fill: 'var(--ink)' }}>{l}</text>)}
        {values.map((row, r) => (
          <g key={r} onClick={onRow ? () => onRow(r) : undefined} style={onRow ? { cursor: 'pointer' } : undefined}>
            {rowLabels[r] !== undefined && <text x={lw / 2} y={lh + (r + 0.5) * cell + 4} text-anchor="middle" style={{ fontSize: '11px', fill: 'var(--ink)' }}>{rowLabels[r]}</text>}
            {row.map((v, c) => <rect key={c} x={lw + c * cell} y={lh + r * cell} width={cell - 1} height={cell - 1} fill={color(v)} />)}
            {showNumbers && cell >= 28 && row.map((v, c) => <text key={'n' + c} x={lw + (c + 0.5) * cell} y={lh + (r + 0.5) * cell + 3} text-anchor="middle" style={{ fontSize: '8px', fill: 'var(--ink)' }}>{v.toFixed(2)}</text>)}
          </g>
        ))}
        {highlightRow !== undefined && highlightRow < rows && <rect x={lw - 1} y={lh + highlightRow * cell - 1} width={cols * cell + 1} height={cell + 1} fill="none" stroke="var(--warn)" stroke-width="2" rx="2" />}
      </svg>
    </div>
  )
}

export function Legend({ items }: { items: { name: string; color: string }[] }): JSX.Element {
  return <div class="legend">{items.map((s) => <span key={s.name}><i style={{ background: s.color }} />{s.name}</span>)}</div>
}
