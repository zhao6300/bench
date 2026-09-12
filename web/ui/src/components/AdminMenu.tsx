import { useEffect, useRef, useState } from "react";

export type AdminAction = "profile-details" | "profile-password" | "profile-avatar" | "logout";

interface AdminMenuProps {
  display: string;
  avatarUrl?: string | null;
  username?: string;
  onAction: (action: AdminAction) => void;
}

const items: { action: AdminAction; label: string; danger?: boolean }[] = [
  { action: "profile-details", label: "修改资料" },
  { action: "profile-avatar", label: "修改头像" },
  { action: "profile-password", label: "修改密码" },
  { action: "logout", label: "退出登录", danger: true },
];

export default function AdminMenu({ display, avatarUrl, username, onAction }: AdminMenuProps) {
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handlePointerDown(event: MouseEvent) {
      if (open && menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }

    document.addEventListener("mousedown", handlePointerDown);
    return () => document.removeEventListener("mousedown", handlePointerDown);
  }, [open]);

  const fallbackLetter = (display || username || "A").slice(0, 1).toUpperCase();

  return (
    <div className="admin-menu" ref={menuRef}>
      <button
        type="button"
        className="admin-menu-button"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((previous) => !previous)}
      >
        {avatarUrl ? <img className="admin-avatar" src={avatarUrl} alt="" /> : <span className="admin-avatar">{fallbackLetter}</span>}
        <span className="admin-label">{display}</span>
      </button>
      {open ? (
        <div className="admin-menu-panel" role="menu">
          {items.map((item) => (
            <button
              key={item.action}
              type="button"
              role="menuitem"
              className={item.danger ? "admin-menu-item admin-menu-item-danger" : "admin-menu-item"}
              onClick={() => {
                setOpen(false);
                onAction(item.action);
              }}
            >
              {item.label}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
