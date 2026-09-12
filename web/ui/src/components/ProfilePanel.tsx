import { useEffect, useState } from "react";
import {
  changePassword,
  fetchProfile,
  type AuthProfile,
  type UpdateProfile,
  updateProfile,
} from "../services";

interface ProfilePanelProps {
  page: "details" | "password" | "avatar";
  onClose: () => void;
  onPasswordChanged: () => void;
  onProfileSaved: (profile: AuthProfile) => void;
}

const MAX_AVATAR_BYTES = 128 * 1024;

async function readAvatarDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error("头像图片读取失败"));
    reader.readAsDataURL(file);
  });
}

export default function ProfilePanel({ page, onClose, onPasswordChanged, onProfileSaved }: ProfilePanelProps) {
  const [profile, setProfile] = useState<AuthProfile | null>(null);
  const [passwordMessage, setPasswordMessage] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [currentPassword, setCurrentPassword] = useState("");
  const [email, setEmail] = useState("");
  const [display, setDisplay] = useState("");
  const [avatarUrl, setAvatarUrl] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [savingProfile, setSavingProfile] = useState(false);
  const [changingPassword, setChangingPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const loaded = await fetchProfile();
        if (cancelled) return;
        setProfile(loaded);
        setDisplay(loaded.display_name ?? "");
        setEmail(loaded.email ?? "");
        setAvatarUrl(loaded.avatar_url ?? "");
      } catch (failure) {
        if (!cancelled) setError(failure instanceof Error ? failure.message : "读取管理员资料失败");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleAvatarChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (file.size > MAX_AVATAR_BYTES) {
      setError("头像文件不能超过 128 KB");
      return;
    }
    try {
      const dataUrl = await readAvatarDataUrl(file);
      setAvatarUrl(dataUrl);
      setError("");
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "头像图片读取失败");
    } finally {
      event.target.value = "";
    }
  }

  async function handleProfileSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (savingProfile) return;
    setSavingProfile(true);
    setError("");
    try {
      const saved = await updateProfile({
        display_name: display,
        email,
        avatar_url: avatarUrl || null,
      } satisfies UpdateProfile);
      setProfile(saved);
      onProfileSaved(saved);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "保存管理员资料失败");
    } finally {
      setSavingProfile(false);
    }
  }

  async function handlePasswordSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (changingPassword) return;
    setChangingPassword(true);
    setError("");
    setPasswordMessage("");
    try {
      if (newPassword.length < 8) throw new Error("新密码长度至少为 8 个字符");
      if (newPassword !== confirmPassword) throw new Error("两次输入的新密码不一致");
      await changePassword({ current_password: currentPassword, new_password: newPassword });
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
      setPasswordMessage("密码已更新");
      onPasswordChanged();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "修改密码失败");
    } finally {
      setChangingPassword(false);
    }
  }

  return (
    <div className="profile-panel">
      <div className="profile-panel-head">
        <h2>{page === "avatar" ? "修改头像" : page === "password" ? "修改密码" : "修改资料"}</h2>
        <button type="button" onClick={onClose} aria-label="返回" className="profile-close">
          返回
        </button>
      </div>

      {loading ? (
        <p>资料加载中…</p>
      ) : (
        <>
          {page === "details" ? (
            <form onSubmit={handleProfileSubmit}>
              <label htmlFor="admin-display-name">显示名称</label>
              <input
                id="admin-display-name"
                value={display}
                onChange={(event) => setDisplay(event.target.value)}
                maxLength={64}
              />
              <label htmlFor="admin-email">邮箱</label>
              <input
                id="admin-email"
                type="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                required
              />
              <button type="submit" disabled={savingProfile}>
                {savingProfile ? "保存中…" : "保存资料"}
              </button>
            </form>
          ) : null}

          {page === "avatar" ? (
            <form onSubmit={(event) => { event.preventDefault(); void handleProfileSubmit(event); }}>
              <div className="avatar-row">
                {avatarUrl ? (
                  <img src={avatarUrl} alt={`${profile?.username || "管理员"} 的头像`} />
                ) : (
                  <span className="avatar-placeholder">{profile?.username?.slice(0, 1).toUpperCase() || "A"}</span>
                )}
              </div>
              <label htmlFor="admin-avatar">{avatarUrl ? "更换头像" : "上传头像"}</label>
              <input
                id="admin-avatar"
                type="file"
                accept="image/png,image/jpeg,image/webp"
                onChange={handleAvatarChange}
              />
              <button type="submit" disabled={savingProfile}>
                {savingProfile ? "保存中…" : "保存头像"}
              </button>
            </form>
          ) : null}

          {page === "password" ? (
            <form onSubmit={handlePasswordSubmit}>
              <label htmlFor="current-password">当前密码</label>
              <input
                id="current-password"
                type="password"
                autoComplete="current-password"
                value={currentPassword}
                onChange={(event) => setCurrentPassword(event.target.value)}
                required
              />
              <label htmlFor="new-password">新密码</label>
              <input
                id="new-password"
                type="password"
                autoComplete="new-password"
                value={newPassword}
                onChange={(event) => setNewPassword(event.target.value)}
                minLength={8}
                required
              />
              <label htmlFor="confirm-password">确认新密码</label>
              <input
                id="confirm-password"
                type="password"
                autoComplete="new-password"
                value={confirmPassword}
                onChange={(event) => setConfirmPassword(event.target.value)}
                minLength={8}
                required
              />
              <button type="submit" disabled={changingPassword}>
                {changingPassword ? "修改中…" : "修改密码"}
              </button>
              {passwordMessage ? <p className="profile-success">{passwordMessage}</p> : null}
            </form>
          ) : null}
        </>
      )}
      {error ? <p className="auth-error">{error}</p> : null}
    </div>
  );
}
