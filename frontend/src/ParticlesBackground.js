import React, { useCallback } from 'react';
import Particles from 'react-tsparticles';
import { loadSlim } from 'tsparticles-slim';

const ParticlesBackground = () => {
    const particlesInit = useCallback(async (engine) => {
        await loadSlim(engine);
    }, []);

    const options = {
        background: {
            color: {
                value: "#121212", // Match our new dark background
            },
        },
        fpsLimit: 60,
        interactivity: {
            events: {
                onHover: {
                    enable: true,
                    mode: "repulse",
                },
                resize: true,
            },
            modes: {
                repulse: {
                    distance: 100,
                    duration: 0.4,
                },
            },
        },
        particles: {
            color: {
                value: "#4a4a7f", // A subtle purple-ish color
            },
            links: {
                color: "#4a4a7f",
                distance: 150,
                enable: true,
                opacity: 0.2,
                width: 1,
            },
            collisions: {
                enable: true,
            },
            move: {
                direction: "none",
                enable: true,
                outModes: {
                    default: "bounce",
                },
                random: false,
                speed: 1,
                straight: false,
            },
            number: {
                density: {
                    enable: true,
                    area: 800,
                },
                value: 50,
            },
            opacity: {
                value: 0.2,
            },
            shape: {
                type: "circle",
            },
            size: {
                value: { min: 1, max: 5 },
            },
        },
        detectRetina: true,
    };

    return (
        <Particles
            id="tsparticles"
            init={particlesInit}
            options={options}
            style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: -1 }}
        />
    );
};

export default ParticlesBackground;