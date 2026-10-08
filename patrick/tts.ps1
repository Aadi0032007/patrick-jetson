param([string]$TextPath, [string]$WavePath)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$patrickSynth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $patrickSynth.SetOutputToWaveFile($WavePath)
    $patrickSynth.Speak([System.IO.File]::ReadAllText($TextPath))
} finally {
    $patrickSynth.Dispose()
}
