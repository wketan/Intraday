/* Intraday Console — service worker (notifications only, no asset caching) */
self.addEventListener('install', (e) => { self.skipWaiting(); });
self.addEventListener('activate', (e) => { e.waitUntil(self.clients.claim()); });

// Focus / open the app when a notification is tapped
self.addEventListener('notificationclick', (e) => {
  e.notification.close();
  e.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then((cs) => {
      for (const c of cs) { if ('focus' in c) return c.focus(); }
      if (self.clients.openWindow) return self.clients.openWindow('./console.html');
    })
  );
});

// Web Push (used only if a server-side push sender is added later)
self.addEventListener('push', (e) => {
  let d = { title: 'New signal', body: 'A trade was found' };
  try { if (e.data) d = e.data.json(); } catch (_) {}
  e.waitUntil(
    self.registration.showNotification(d.title || 'New signal', {
      body: d.body || '', icon: 'icon-192.png', badge: 'icon-192.png',
      tag: 'ie-signal', renotify: true
    })
  );
});
