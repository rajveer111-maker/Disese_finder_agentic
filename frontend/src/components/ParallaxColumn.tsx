'use client';

import React, { useEffect, useState } from 'react';
import { motion, useScroll, useTransform, useSpring } from 'framer-motion';

interface ParallaxColumnProps {
  children: React.ReactNode;
  speed?: number; // Pixel translation distance at maximum scroll
  className?: string;
}

export default function ParallaxColumn({ children, speed = -60, className = '' }: ParallaxColumnProps) {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768);
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const { scrollYProgress } = useScroll();
  // Smooth scroll progress with custom spring stiffness and damping
  const smoothProgress = useSpring(scrollYProgress, { stiffness: 400, damping: 90 });
  
  // Translate columns upwards or downwards depending on scroll
  const y = useTransform(smoothProgress, [0, 1], [0, isMobile ? 0 : speed]);

  return (
    <motion.div style={{ y }} className={className}>
      {children}
    </motion.div>
  );
}
