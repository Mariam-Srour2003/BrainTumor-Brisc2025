document.getElementById('loadBtn').addEventListener('click', ()=>{
  const path = document.getElementById('imagePath').value.trim();
  const overlay = document.getElementById('overlay');
  const heatmap = document.getElementById('heatmap');
  const meta = document.getElementById('meta');
  overlay.src = path;
  heatmap.src = path.replace('.png','_heatmap.png');
  meta.innerText = `Loaded: ${path}`;
});
// auto-load
window.addEventListener('load', ()=>document.getElementById('loadBtn').click());