(function () {
  if (typeof TrackUI === 'undefined' || typeof TrackUI.record !== 'function') return;
  TrackUI.record({
    postServer: 'https://api.thetracehub.com',
    regularEvents: '*',
    pollingEvents: '*',
    pollingMs: 100,
    debug: false,
    postInterval: 1
  });
})();
