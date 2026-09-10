interface StatusBadgeProps {
  status?: string;
}

export default function StatusBadge({ status }: StatusBadgeProps) {
  return <span className={`status-badge ${status ?? "unknown"}`}>{status ?? "unknown"}</span>;
}
