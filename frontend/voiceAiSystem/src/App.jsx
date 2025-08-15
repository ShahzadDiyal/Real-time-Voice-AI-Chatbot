import { useState } from 'react'
import VoiceAiSystem from './components/VoiceAiSystem'
function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <VoiceAiSystem />
    </>
  )
}

export default App
