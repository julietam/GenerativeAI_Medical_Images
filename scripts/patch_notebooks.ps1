param()
$ErrorActionPreference = 'Stop'

function Update-Pix2Pix($Path) {
  $nb = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
  foreach ($cell in $nb.cells) {
    if ($cell.cell_type -ne 'code') { continue }
    $text = ($cell.source -join '')
    if ($text -like "*root = 'data/paired_mri'*") {
      $lines = $text -split "\r?\n"
      for ($i=0; $i -lt $lines.Length; $i++) {
        if ($lines[$i] -match "^root = 'data/paired_mri'") {
          $lines[$i] = "root = 'data/sample_mri_pairs'"
          $block = @(
            "# Autogenera datos de muestra si no existen",
            "from pathlib import Path",
            "import subprocess, sys",
            "if not any((Path(root)/'T1').glob('*.png')) or not any((Path(root)/'T2').glob('*.png')):",
            "    try:",
            "        subprocess.run([sys.executable, 'scripts/make_sample_data.py'], check=False)",
            "    except Exception as e:",
            "        print('Warning: could not generate sample data:', e)"
          )
          $lines = @($lines[0..$i] + $block + $lines[($i+1)..($lines.Length-1)])
          break
        }
      }
      $cell.source = @(($lines -join "`n"))
    }
  }
  ($nb | ConvertTo-Json -Depth 100) | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Update-CycleGAN($Path) {
  $nb = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
  foreach ($cell in $nb.cells) {
    if ($cell.cell_type -ne 'code') { continue }
    $text = ($cell.source -join '')
    if ($text -like "*rootA, rootB = 'data/unpaired_mri', 'data/unpaired_ct'*") {
      $lines = $text -split "\r?\n"
      for ($i=0; $i -lt $lines.Length; $i++) {
        if ($lines[$i] -match "^rootA, rootB = 'data/unpaired_mri', 'data/unpaired_ct'") {
          $lines[$i] = "rootA, rootB = 'data/sample_unpaired_mri', 'data/sample_unpaired_ct'"
          $block = @(
            "# Autogenera datos de muestra si no existen",
            "from pathlib import Path",
            "import subprocess, sys",
            "if not any(Path(rootA).glob('*.png')) or not any(Path(rootB).glob('*.png')):",
            "    try:",
            "        subprocess.run([sys.executable, 'scripts/make_sample_data.py'], check=False)",
            "    except Exception as e:",
            "        print('Warning: could not generate sample data:', e)"
          )
          $lines = @($lines[0..$i] + $block + $lines[($i+1)..($lines.Length-1)])
          break
        }
      }
      $cell.source = @(($lines -join "`n"))
    }
  }
  ($nb | ConvertTo-Json -Depth 100) | Set-Content -LiteralPath $Path -Encoding UTF8
}

Update-Pix2Pix -Path "C:\Users\julie\GenerativeAI_Medical_Images\03_pix2pix_mri.ipynb"
Update-CycleGAN -Path "C:\Users\julie\GenerativeAI_Medical_Images\04_cyclegan_ct_mri.ipynb"
Write-Output "Patched notebooks."
