'use client';

import dynamic from 'next/dynamic';

const OnnxTestBarComponent = dynamic(() => import('../components/onnx-test-bar'), { ssr: false });

export default function Home() {
  return (
    <div>
      <main>
        <OnnxTestBarComponent />
      </main>
    </div>
  );
}
