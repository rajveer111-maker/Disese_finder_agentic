'use client';

import React, { useRef } from 'react';
import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion';

interface Card3DProps {
  children: React.ReactNode;
  className?: string;
  glowColor?: string;
}

export default function Card3D({ children, className = '', glowColor = 'rgba(139, 92, 246, 0.25)' }: Card3DProps) {
  const ref = useRef<HTMLDivElement>(null);
  
  // Motion values to track mouse coordinate offsets relative to center
  const x = useMotionValue(0);
  const y = useMotionValue(0);
  
  // Spring configurations for physics-based response
  const rotateX = useSpring(useTransform(y, [-0.5, 0.5], [10, -10]), { stiffness: 200, damping: 15 });
  const rotateY = useSpring(useTransform(x, [-0.5, 0.5], [-10, 10]), { stiffness: 200, damping: 15 });
  const scale = useSpring(1, { stiffness: 200, damping: 15 });

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!ref.current) return;
    
    const rect = ref.current.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    
    // Distance from the mouse to the center of the element
    const mouseX = e.clientX - rect.left - width / 2;
    const mouseY = e.clientY - rect.top - height / 2;
    
    // Map to normalized range [-0.5, 0.5]
    x.set(mouseX / width);
    y.set(mouseY / height);
    
    scale.set(1.05);
  };

  const handleMouseLeave = () => {
    x.set(0);
    y.set(0);
    scale.set(1);
  };

  // Split classes to apply flex layout and padding classes to the inner content wrapper
  // and outer box/sizing classes to the outer motion wrapper.
  const classes = className.split(' ').filter(Boolean);
  
  const isPaddingClass = (c: string) => {
    const clean = c.includes(':') ? c.split(':').pop() || '' : c;
    return (
      clean.startsWith('p-') ||
      clean.startsWith('px-') ||
      clean.startsWith('py-') ||
      clean.startsWith('pl-') ||
      clean.startsWith('pr-') ||
      clean.startsWith('pt-') ||
      clean.startsWith('pb-')
    );
  };

  const parentClasses = classes.filter(c => 
    !c.startsWith('flex') && 
    !c.startsWith('justify-') && 
    !c.startsWith('items-') &&
    !isPaddingClass(c)
  ).join(' ');
  
  const contentClasses = classes.filter(c => 
    c.startsWith('flex') || 
    c.startsWith('justify-') || 
    c.startsWith('items-')
  ).join(' ');

  return (
    <motion.div
      ref={ref}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{
        rotateX,
        rotateY,
        scale,
        transformStyle: 'preserve-3d',
      }}
      className={`relative glass-panel transition-all duration-300 hover:bg-white/[0.05] hover:border-primary-purple/40 ${parentClasses}`}
    >
      {/* Background glow overlay */}
      <div
        className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none -z-10"
        style={{
          background: `radial-gradient(circle at center, ${glowColor}, transparent 70%)`,
          filter: 'blur(20px)',
        }}
      />
      
      {/* Dynamic light reflection shimmer */}
      <div 
        className="absolute inset-0 rounded-2xl bg-gradient-to-tr from-white/0 via-white/5 to-white/0 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-500" 
        style={{ transform: 'translateZ(10px)' }}
      />
      
      {/* Content wrapper with depth projection */}
      <div 
        style={{ transform: 'translateZ(25px)' }} 
        className={`h-full w-full relative z-10 card-content-wrapper ${contentClasses}`}
      >
        {children}
      </div>
    </motion.div>
  );
}
