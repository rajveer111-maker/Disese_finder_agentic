'use client';

import { useEffect, useState } from 'react';
import { useMotionValue, useSpring, motion } from 'framer-motion';

export default function MouseGlow() {
  const mouseX = useMotionValue(-500);
  const mouseY = useMotionValue(-500);
  const [mounted, setMounted] = useState(false);

  const glowX = useSpring(mouseX, { stiffness: 80, damping: 20 });
  const glowY = useSpring(mouseY, { stiffness: 80, damping: 20 });

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMounted(true);
    const handleMouseMove = (e: MouseEvent) => {
      // Offset by half of the width/height (200px)
      mouseX.set(e.clientX - 200);
      mouseY.set(e.clientY - 200);
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, [mouseX, mouseY]);

  if (!mounted) return null;

  return (
    <motion.div
      className="pointer-events-none fixed z-10 w-[400px] h-[400px] rounded-full blur-[100px] opacity-40 mix-blend-screen bg-gradient-to-tr from-primary-purple/40 via-accent-cyan/15 to-transparent hidden md:block"
      style={{
        left: glowX,
        top: glowY,
      }}
    />
  );
}
