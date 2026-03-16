/**
 * PolyMind AI — 3D Polymer Visualization
 * Author: Ansh Sharma | B230825MT
 */
import { useEffect, useRef } from 'react';
import * as THREE from 'three';

export default function Polymer3D({ stress = 0.1, dark = true, height = 320 }) {
  const mountRef  = useRef(null);
  const stressRef = useRef(stress);
  stressRef.current = stress;

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const W = mount.clientWidth || 440;
    const H = height;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(W, H);
    renderer.setClearColor(0x000000, 0);
    mount.appendChild(renderer.domElement);

    const scene  = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(55, W / H, 0.1, 1000);
    camera.position.set(0, 2, 12);
    camera.lookAt(0, 0, 0);

    scene.fog = new THREE.FogExp2(dark ? 0x060611 : 0xf0f4ff, 0.022);
    scene.add(new THREE.AmbientLight(0x334466, 3));
    const dir = new THREE.DirectionalLight(0x00d4ff, 5);
    dir.position.set(8, 10, 8); scene.add(dir);
    const pt1 = new THREE.PointLight(0x9d4edd, 4, 25);
    pt1.position.set(-6, 4, 2); scene.add(pt1);
    const pt2 = new THREE.PointLight(0xf72585, 2, 20);
    pt2.position.set(6, -4, -2); scene.add(pt2);

    const ATOMS = [
      { color: 0x00d4ff, r: 0.30 },
      { color: 0xffee44, r: 0.38 },
      { color: 0xff4466, r: 0.26 },
      { color: 0x44ff88, r: 0.28 },
      { color: 0x00d4ff, r: 0.30 },
    ];

    const N = 26, atoms = [];
    for (let i = 0; i < N; i++) {
      const def = ATOMS[i % 5];
      const bx  = (i - N / 2) * 0.72;
      const by  = Math.sin(i * 0.82) * 0.9;
      const bz  = Math.cos(i * 0.52) * 0.5;
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(def.r, 16, 16),
        new THREE.MeshPhongMaterial({ color: def.color, emissive: def.color, emissiveIntensity: 0.2, shininess: 100 })
      );
      mesh.position.set(bx, by, bz);
      scene.add(mesh);
      atoms.push({ mesh, bx, by, bz });
      if (i > 0) {
        const prev = atoms[i - 1];
        const sv = new THREE.Vector3(prev.bx, prev.by, prev.bz);
        const ev = new THREE.Vector3(bx, by, bz);
        const d  = ev.clone().sub(sv);
        const bond = new THREE.Mesh(
          new THREE.CylinderGeometry(0.06, 0.06, d.length(), 8),
          new THREE.MeshPhongMaterial({ color: 0x4488aa, transparent: true, opacity: 0.6 })
        );
        bond.position.copy(sv.clone().add(ev).multiplyScalar(0.5));
        bond.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.clone().normalize());
        scene.add(bond);
      }
    }

    const rings = [];
    [4, 10, 18].forEach(idx => {
      if (idx < atoms.length) {
        const ring = new THREE.Mesh(
          new THREE.TorusGeometry(0.65, 0.02, 8, 40),
          new THREE.MeshBasicMaterial({ color: 0x00d4ff, transparent: true, opacity: 0.3 })
        );
        ring.position.copy(atoms[idx].mesh.position);
        scene.add(ring);
        rings.push({ mesh: ring, atom: atoms[idx] });
      }
    });

    const pN   = 180;
    const pGeo = new THREE.BufferGeometry();
    const pPos = new Float32Array(pN * 3);
    for (let i = 0; i < pN; i++) {
      pPos[i * 3]     = (Math.random() - 0.5) * 22;
      pPos[i * 3 + 1] = (Math.random() - 0.5) * 8;
      pPos[i * 3 + 2] = (Math.random() - 0.5) * 6;
    }
    pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
    const particles = new THREE.Points(pGeo,
      new THREE.PointsMaterial({ color: 0x00d4ff, size: 0.08, transparent: true, opacity: 0.5 })
    );
    scene.add(particles);
    const grid = new THREE.GridHelper(26, 26, 0x112233, 0x0a1020);
    grid.position.y = -2.8; scene.add(grid);

    let frame = 0, raf;
    const animate = () => {
      raf = requestAnimationFrame(animate);
      frame++;
      const t = frame * 0.016;
      const s = stressRef.current;
      atoms.forEach(({ mesh, bx, by, bz }, i) => {
        mesh.position.x = bx * (1 + s * 0.25);
        mesh.position.y = by + s * 0.2 * Math.sin(t * 4.5 + i * 0.65);
        mesh.position.z = bz + s * 0.12 * Math.cos(t * 3 + i * 0.4);
        mesh.rotation.y += 0.01;
        mesh.material.emissiveIntensity = 0.2 + s * 0.4 * Math.abs(Math.sin(t * 2 + i));
      });
      const pArr = particles.geometry.attributes.position.array;
      for (let j = 0; j < pN; j++) {
        pArr[j * 3 + 2] -= 0.022 * (1 + s * 2);
        if (pArr[j * 3 + 2] < -10) pArr[j * 3 + 2] = 10;
      }
      particles.geometry.attributes.position.needsUpdate = true;
      rings.forEach(({ mesh: ring, atom }, i) => {
        ring.position.copy(atom.mesh.position);
        ring.rotation.x += 0.014 + i * 0.003;
        ring.rotation.z += 0.009 + i * 0.002;
        ring.material.opacity = 0.2 + s * 0.3 + 0.1 * Math.sin(t * 2 + i);
      });
      scene.rotation.y = Math.sin(t * 0.15) * 0.45;
      scene.rotation.x = Math.sin(t * 0.08) * 0.07;
      pt1.intensity = 3 + s * 3 + Math.sin(t * 2) * 1.5;
      renderer.render(scene, camera);
    };
    animate();

    const onResize = () => {
      const W2 = mount.clientWidth;
      camera.aspect = W2 / H;
      camera.updateProjectionMatrix();
      renderer.setSize(W2, H);
    };
    window.addEventListener('resize', onResize);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
      renderer.dispose();
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement);
    };
  }, [height, dark]);

  return (
    <div style={{ position: 'relative' }}>
      <div ref={mountRef} style={{
        width: '100%', height, borderRadius: 12, overflow: 'hidden',
        background: dark
          ? 'radial-gradient(ellipse at center,#0a0a1f 0%,#060611 100%)'
          : 'radial-gradient(ellipse at center,#e8eaff 0%,#f0f4ff 100%)',
      }} />
      <div style={{ display: 'flex', justifyContent: 'center', gap: 14, marginTop: 8, fontSize: 11, color: dark ? '#7777aa' : '#6666aa' }}>
        <span style={{ color: '#00d4ff' }}>&#9679; C</span>
        <span style={{ color: '#ffee44' }}>&#9679; S</span>
        <span style={{ color: '#ff4466' }}>&#9679; O</span>
        <span style={{ color: '#44ff88' }}>&#9679; N</span>
        <span style={{ color: 'rgba(0,212,255,.45)' }}>&#183; e&#8315;</span>
      </div>
    </div>
  );
}
