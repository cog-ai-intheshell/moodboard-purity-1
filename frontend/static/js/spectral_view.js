// Spectral Aesthetic Analysis rendering helpers.
// The graph runtime calls these after `/api/analyze` returns Laplacian metrics.

      function buildHarmonySpectrum() {
        var spectral = analysisPayload && analysisPayload.spectralAnalysis ? analysisPayload.spectralAnalysis : null;
        if (!spectral || !spectral.eigenvalues || !spectral.eigenvalues.length) return null;
        var eigenvalues = spectral.eigenvalues.map(Number);
        var energy = spectral.energy && spectral.energy.length ? spectral.energy.map(Number) : eigenvalues.map(function(value) {
          var total = eigenvalues.reduce(function(sum, item) { return sum + Math.max(0, item); }, 0);
          return total > 0 ? Math.max(0, value) / total : 0;
        });
        var maxEigenvalue = Math.max.apply(null, eigenvalues.concat([1e-9]));
        var points = [];
        var peak = 0;
        var peakIndex = 0;
        for (var i = 0; i < 128; i += 1) {
          var x = i / 127;
          var value = 0;
          eigenvalues.forEach(function(eigenvalue, index) {
            var position = eigenvalues.length <= 1 ? 0 : index / (eigenvalues.length - 1);
            var amplitude = (energy[index] || 0) * 7.5 + (eigenvalue / maxEigenvalue) * 0.08;
            var width = 0.020 + Math.min(0.026, 0.16 / Math.max(8, eigenvalues.length));
            var distance = x - position;
            value += amplitude * Math.exp(-(distance * distance) / (2 * width * width));
          });
          value = Math.max(0, value);
          if (value > peak) {
            peak = value;
            peakIndex = i;
          }
          points.push(value);
        }
        return { points: points, peak: peak, peakIndex: peakIndex, spectral: spectral };
      }

      function updateSpectralMetrics(spectral) {
        var empty = !spectral || !spectral.eigenvalues || !spectral.eigenvalues.length;
        document.getElementById("spectralHarmonicity").textContent = empty ? "--" : formatPercent(spectral.harmonicityScore);
        document.getElementById("spectralMoodPurity").textContent = empty ? "--" : formatPercent(spectral.purityScore);
        document.getElementById("spectralPurity").textContent = empty ? "--" : formatPercent(spectral.spectralPurityScore === undefined ? spectral.purityScore : spectral.spectralPurityScore);
        document.getElementById("spectralFrequency").textContent = empty ? "--" : (firstMetricValue(spectral.dominantAestheticFrequency) || 0).toFixed(2) + "f";
        document.getElementById("spectralGap").textContent = empty ? "--" : (firstMetricValue(spectral.normalizedSpectralGap, spectral.spectralGap) || 0).toFixed(3);
        document.getElementById("spectralRegimes").textContent = empty ? "--" : String(spectral.aestheticRegimeCount || 0);
        document.getElementById("spectralDissonance").textContent = empty ? "--" : formatPercent(spectral.dissonanceScore);
      }

      function drawEmptyHarmony(width, height) {
        var padding = 14;
        harmonyCtx.save();
        harmonyCtx.clearRect(0, 0, width, height);
        harmonyCtx.fillStyle = cssVar("--field-bg", "#191919");
        harmonyCtx.fillRect(0, 0, width, height);
        harmonyCtx.strokeStyle = cssVar("--canvas-grid", "rgba(248,248,248,0.08)");
        harmonyCtx.lineWidth = 1;
        harmonyCtx.globalAlpha = 0.55;
        for (var lineIndex = 0; lineIndex <= 3; lineIndex += 1) {
          var gridY = padding + ((height - padding * 2) * lineIndex) / 3;
          harmonyCtx.beginPath();
          harmonyCtx.moveTo(padding, gridY);
          harmonyCtx.lineTo(width - padding, gridY);
          harmonyCtx.stroke();
        }
        harmonyCtx.globalAlpha = 1;
        harmonyCtx.fillStyle = cssVar("--muted", "#4b5563");
        harmonyCtx.font = "12px Inter, system-ui, sans-serif";
        harmonyCtx.textAlign = "center";
        harmonyCtx.fillText(files.length ? "Spectral graph pending." : "Upload images to compute graph spectrum.", width / 2, height / 2);
        harmonyCtx.restore();
      }

      function drawHarmony() {
        var rect = harmonyCanvas.getBoundingClientRect();
        if (!rect.width || !rect.height) return;
        var spectrum = buildHarmonySpectrum();
        var width = rect.width;
        var height = rect.height;
        var padding = 14;
        var chartWidth = Math.max(1, width - padding * 2);
        var chartHeight = Math.max(1, height - padding * 2);
        if (!spectrum) {
          updateSpectralMetrics(null);
          document.getElementById("harmonyStatus").textContent = files.length ? "Waiting for graph Laplacian eigenvalues." : "Waiting for graph spectrum.";
          drawEmptyHarmony(width, height);
          return;
        }
        updateSpectralMetrics(spectrum.spectral);
        var maxValue = Math.max(spectrum.peak, 1e-9);
        var peakFrequency = spectrum.peakIndex / Math.max(1, spectrum.points.length - 1);
        document.getElementById("harmonyStatus").textContent = (spectrum.spectral.interpretation || "Laplacian spectrum ready.") + " Peak " + peakFrequency.toFixed(2) + "f.";

        harmonyCtx.save();
        harmonyCtx.clearRect(0, 0, width, height);
        harmonyCtx.fillStyle = cssVar("--field-bg", "#191919");
        harmonyCtx.fillRect(0, 0, width, height);
        harmonyCtx.strokeStyle = cssVar("--canvas-grid", "rgba(248,248,248,0.08)");
        harmonyCtx.lineWidth = 1;
        harmonyCtx.globalAlpha = 0.65;
        for (var lineIndex = 0; lineIndex <= 3; lineIndex += 1) {
          var gridY = padding + (chartHeight * lineIndex) / 3;
          harmonyCtx.beginPath();
          harmonyCtx.moveTo(padding, gridY);
          harmonyCtx.lineTo(width - padding, gridY);
          harmonyCtx.stroke();
        }
        harmonyCtx.globalAlpha = 1;

        harmonyCtx.fillStyle = colorToRgba(cssVar("--canvas-chart", "#f89540"), 0.09);
        harmonyCtx.beginPath();
        for (var i = 0; i < spectrum.points.length; i += 1) {
          var x = padding + (chartWidth * i) / Math.max(1, spectrum.points.length - 1);
          var curveY = padding + chartHeight - (chartHeight * spectrum.points[i]) / maxValue;
          if (i === 0) harmonyCtx.moveTo(x, padding + chartHeight);
          harmonyCtx.lineTo(x, curveY);
        }
        harmonyCtx.lineTo(width - padding, padding + chartHeight);
        harmonyCtx.closePath();
        harmonyCtx.fill();

        harmonyCtx.strokeStyle = cssVar("--canvas-chart", "#f89540");
        harmonyCtx.lineWidth = 2;
        harmonyCtx.beginPath();
        for (var j = 0; j < spectrum.points.length; j += 1) {
          var px = padding + (chartWidth * j) / Math.max(1, spectrum.points.length - 1);
          var py = padding + chartHeight - (chartHeight * spectrum.points[j]) / maxValue;
          if (j === 0) harmonyCtx.moveTo(px, py);
          else harmonyCtx.lineTo(px, py);
        }
        harmonyCtx.stroke();

        var peakX = padding + chartWidth * peakFrequency;
        harmonyCtx.strokeStyle = colorToRgba(cssVar("--canvas-electron", "#5D71FC"), 0.56);
        harmonyCtx.lineWidth = 1;
        harmonyCtx.beginPath();
        harmonyCtx.moveTo(peakX, padding);
        harmonyCtx.lineTo(peakX, padding + chartHeight);
        harmonyCtx.stroke();
        harmonyCtx.fillStyle = cssVar("--muted", "#4b5563");
        harmonyCtx.font = "11px Inter, system-ui, sans-serif";
        harmonyCtx.textAlign = "left";
        harmonyCtx.fillText("0", padding, height - 8);
        harmonyCtx.textAlign = "right";
        harmonyCtx.fillText("1.0 f", width - padding, height - 8);
        var peakLabelX = Math.min(width - padding - 4, peakX + 6);
        harmonyCtx.textAlign = peakLabelX > width - 90 ? "right" : "left";
        harmonyCtx.fillText("peak", peakLabelX, padding + 20);
        harmonyCtx.restore();
      }
