// Main browser runtime for the local Moodboard app.
// This file was extracted from moodboard_interface.html so the frontend can be split in tested slices.

      var files = [];
      var objectUrls = [];
      var previewTimer = null;
      var previewAbort = null;
      var previewGeneration = 0;
      var analysisTimer = null;
      var analysisAbort = null;
      var analysisGeneration = 0;
      var analysisPayload = null;
      var analysisTagGroups = [];
      var activeView = "bento";
      var customCells = [];
      var customRects = [];
      var defaultCustomRects = [
        { x: 0, y: 0, w: 2, h: 2 },
        { x: 0, y: 2, w: 1, h: 2 },
        { x: 1, y: 2, w: 1, h: 2 },
        { x: 2, y: 0, w: 2, h: 1 },
        { x: 4, y: 0, w: 2, h: 1 },
        { x: 4, y: 1, w: 1, h: 2 },
        { x: 5, y: 1, w: 1, h: 2 },
        { x: 2, y: 1, w: 2, h: 1 },
        { x: 2, y: 2, w: 2, h: 1 },
        { x: 2, y: 3, w: 1, h: 1 },
        { x: 3, y: 3, w: 1, h: 1 },
        { x: 4, y: 3, w: 2, h: 1 }
      ];
      var customPalette = ["#5D71FC", "#e87511", "#27AE60", "#EB5757", "#1677e8", "#1b998b", "#c5d86d", "#f46036"];
      var presetSizes = {
        a4_landscape: [3508, 2480],
        a4_portrait: [2480, 3508],
        screen_16_9: [4096, 2304],
        iphone: [393, 852],
        custom: [3508, 2480]
      };
      var presetLabels = {
        a4_landscape: "A4 landscape",
        a4_portrait: "A4 portrait",
        screen_16_9: "16:9 landscape",
        iphone: "iPhone",
        custom: "Custom"
      };

      var imageInput = document.getElementById("imageInput");
      var dropZone = document.getElementById("dropZone");
      var clearButton = document.getElementById("clearButton");
      var generateButton = document.getElementById("generateButton");
      var fullscreenPreview = document.getElementById("fullscreenPreview");
      var statusLine = document.getElementById("statusLine");
      var fileStatus = document.getElementById("fileStatus");
      var previewPages = document.getElementById("previewPages");
      var emptyPreview = document.getElementById("emptyPreview");
      var thumbGrid = document.getElementById("thumbGrid");
      var thumbGridNav = document.getElementById("thumbGridNav");
      var customGrid = document.getElementById("customGrid");
      var customGridPanel = document.getElementById("customGridPanel");
      var manualImagesPerPage = Number(document.getElementById("imagesPerPage").value);
      var graphCanvas = document.getElementById("graphCanvas");
      var graphCtx = graphCanvas.getContext("2d");
      var graphFrame = document.getElementById("graphFrame");
      var graphTooltip = document.getElementById("graphTooltip");
      var graphEmpty = document.getElementById("graphEmpty");
      var harmonyCanvas = document.getElementById("harmonyCanvas");
      var harmonyCtx = harmonyCanvas.getContext("2d");
      var graphAnimationFrame = null;
      var graphState = {
        zoom: 1,
        panX: 0,
        panY: 0,
        motionTime: 0,
        motionEnabled: true,
        dragging: false,
        draggingNode: null,
        dragStartX: 0,
        dragStartY: 0,
        panStartX: 0,
        panStartY: 0,
        hoverNodeId: null,
        selectedNodeId: null,
        typeFilter: "all",
        clusterFilter: "all",
        minEdge: 0.05,
        showLabels: true,
        atomLayout: false
      };
      var graphColors = ["#5D71FC", "#EB5757", "#f89540", "#27AE60", "#A855F7", "#F2C94C", "#56CCF2", "#FF6FB1"];

      function setStatus(message) {
        statusLine.textContent = message;
      }

      function rectsToCells(rects) {
        var cells = new Array(24).fill(0);
        rects.forEach(function(rect, rectIndex) {
          for (var y = rect.y; y < rect.y + rect.h; y += 1) {
            for (var x = rect.x; x < rect.x + rect.w; x += 1) {
              cells[y * 6 + x] = rectIndex % customPalette.length;
            }
          }
        });
        return cells;
      }

      function cellsToRects(cells) {
        var visited = new Set();
        var rects = [];
        // The custom grid is painted cell-by-cell, then adjacent cells with the same color become one bento block.
        for (var y = 0; y < 4; y += 1) {
          for (var x = 0; x < 6; x += 1) {
            var start = y * 6 + x;
            if (visited.has(start)) continue;
            var color = cells[start];
            var width = 1;
            while (x + width < 6 && cells[y * 6 + x + width] === color && !visited.has(y * 6 + x + width)) {
              width += 1;
            }
            var height = 1;
            var canGrow = true;
            while (y + height < 4 && canGrow) {
              for (var xx = x; xx < x + width; xx += 1) {
                var idx = (y + height) * 6 + xx;
                if (cells[idx] !== color || visited.has(idx)) {
                  canGrow = false;
                  break;
                }
              }
              if (canGrow) height += 1;
            }
            for (var yy = y; yy < y + height; yy += 1) {
              for (var markX = x; markX < x + width; markX += 1) {
                visited.add(yy * 6 + markX);
              }
            }
            rects.push({ x: x, y: y, w: width, h: height });
          }
        }
        return rects;
      }

      function syncRectsFromCells() {
        customRects = cellsToRects(customCells);
      }

      function formatRangeValue(input) {
        var value = Number(input.value || 0);
        if (input.id === "orientationThreshold") return value.toFixed(2);
        return String(Math.round(value));
      }

      function updateRange(input, displayValue) {
        var min = Number(input.min || 0);
        var max = Number(input.max || 100);
        var value = Number(input.value || 0);
        var progress = ((value - min) / (max - min)) * 100;
        input.style.setProperty("--range-progress", progress + "%");
        var output = document.getElementById(input.id + "Value");
        if (output) output.value = displayValue === undefined ? formatRangeValue(input) : String(displayValue);
      }

      function updatePresetFields() {
        var preset = document.getElementById("pagePreset").value;
        var widthInput = document.getElementById("pageWidth");
        var heightInput = document.getElementById("pageHeight");
        if (preset !== "custom") {
          widthInput.value = presetSizes[preset][0];
          heightInput.value = presetSizes[preset][1];
          widthInput.disabled = true;
          heightInput.disabled = true;
        } else {
          widthInput.disabled = false;
          heightInput.disabled = false;
        }
        document.getElementById("formatChip").textContent = presetLabels[preset] || "Custom";
      }

      function updateOptimizerState() {
        var enabled = document.getElementById("bentoOptimizer").checked;
        var input = document.getElementById("imagesPerPage");
        var mode = document.getElementById("bentoOptimizerMode");
        if (enabled) {
          manualImagesPerPage = Number(input.value) || manualImagesPerPage;
        } else {
          input.value = String(manualImagesPerPage);
          updateRange(input);
        }
        input.disabled = enabled;
        mode.disabled = !enabled;
        updateMetrics();
      }

      function updateBackgroundFromColor() {
        document.getElementById("backgroundText").value = document.getElementById("backgroundColor").value;
      }

      function updateBackgroundFromText() {
        var value = document.getElementById("backgroundText").value.trim();
        if (/^#[0-9A-Fa-f]{6}$/.test(value)) {
          document.getElementById("backgroundColor").value = value;
        }
      }

      function updateChips() {
        var layout = document.getElementById("layoutMode").value;
        document.getElementById("modeChip").textContent = layout;
        document.getElementById("countChip").textContent = files.length + (files.length > 1 ? " images" : " image");
      }

      function customSlotCount() {
        return customRects.length;
      }

      function optimizerImagesPerPageEstimate() {
        if (!files.length) return 0;
        var mode = document.getElementById("bentoOptimizerMode").value;
        var targets = {
          editorial: 6,
          balanced: 10,
          dense: 16,
          clustered: 9
        };
        var target = targets[mode] || 10;
        var estimate = Math.min(files.length, target);
        if (files.length > target * 1.7) estimate = Math.min(files.length, Math.round(Math.sqrt(files.length) * 3.2));
        return Math.max(1, Math.min(20, estimate));
      }

      function constrainImagesPerPage(value) {
        var layout = document.getElementById("layoutMode").value;
        var perPage = Math.max(0, Math.round(Number(value) || 0));
        if (layout === "grid" || layout === "random") perPage = Math.min(perPage, 12);
        if (layout === "custom") perPage = Math.min(perPage, customSlotCount());
        return perPage;
      }

      function syncImagesPerPageDisplay(perPage) {
        var input = document.getElementById("imagesPerPage");
        var outputValue = Math.max(0, Math.round(Number(perPage) || 0));
        if (document.getElementById("bentoOptimizer").checked) {
          var min = Number(input.min || 1);
          var max = Number(input.max || 30);
          input.value = String(Math.max(min, Math.min(max, outputValue || min)));
        }
        updateRange(input, outputValue);
      }

      function updateMetrics(pageCount, perPageOverride) {
        var optimizer = document.getElementById("bentoOptimizer").checked;
        var selectedPerPage = Number(document.getElementById("imagesPerPage").value);
        var generatedPerPage = optimizer ? optimizerImagesPerPageEstimate() : selectedPerPage;
        var perPage = constrainImagesPerPage(perPageOverride || generatedPerPage);
        var pages = pageCount || (files.length ? Math.ceil(files.length / Math.max(1, perPage)) : 0);
        syncImagesPerPageDisplay(perPage);
        document.getElementById("metricImages").textContent = String(files.length);
        document.getElementById("metricPages").textContent = String(pages);
        document.getElementById("metricPerPage").textContent = String(perPage);
        updateProfilePanel();
        fileStatus.textContent = files.length ? files.length + " image(s) loaded." : "No images loaded.";
        updateChips();
      }

      function updateProfilePanel() {
        var palette = document.getElementById("profilePalette");
        var gradient = document.getElementById("profileGradient");
        if (!analysisPayload || !analysisPayload.scores) {
          document.getElementById("profilePurity").textContent = "--";
          palette.innerHTML = skeletonPaletteHtml();
          gradient.classList.add("is-skeleton");
          gradient.style.background = "";
          document.getElementById("profileColorAnalysis").innerHTML = skeletonColorAnalysisHtml();
          return;
        }
        document.getElementById("profilePurity").textContent = Math.round(analysisPayload.scores.purity * 100) + "%";
        var colors = analysisPayload.palette ? analysisPayload.palette.slice(0, 5) : [];
        if (!colors.length) colors = [];
        palette.innerHTML = colors.map(function(color) {
          return '<span class="palette-chip" style="background:' + escapeHtml(color.hex || "#222222") + '" title="' + escapeHtml(color.name || color.hex || "") + '"></span>';
        }).join("");
        gradient.classList.remove("is-skeleton");
        gradient.style.background = gradientFromColors(colors);
        document.getElementById("profileColorAnalysis").innerHTML = colorAnalysisHtml(colors);
      }

      function gradientFromColors(colors) {
        if (!colors.length) return "var(--field-bg)";
        if (colors.length === 1) return colors[0].hex || "var(--field-bg)";
        return "linear-gradient(90deg, " + colors.map(function(color, index) {
          var stop = Math.round((index / Math.max(1, colors.length - 1)) * 100);
          return (color.hex || "#222222") + " " + stop + "%";
        }).join(", ") + ")";
      }

      function skeletonPaletteHtml() {
        return Array.from({ length: 5 }, function() {
          return '<span class="palette-chip skeleton-chip"></span>';
        }).join("");
      }

      function skeletonColorAnalysisHtml() {
        return Array.from({ length: 5 }, function() {
          return '<div class="color-row-analysis is-skeleton"><i></i><span><b class="skeleton-line"></b></span><strong><b class="skeleton-line short"></b></strong></div>';
        }).join("");
      }

      function colorAnalysisHtml(colors) {
        var roles = ["Dominante", "Primaire", "Secondaire", "Tertiaire", "Accent"];
        return colors.slice(0, 5).map(function(color, index) {
          var weight = Math.round(Number(color.weight || (index === 0 ? 0.42 : 0.14)) * 100);
          var rgb = color.rgb || hexToRgb(color.hex || "#222222");
          var temperature = colorTemperatureLabel(rgb);
          return '<div class="color-row-analysis">' +
            '<i style="background:' + escapeHtml(color.hex || "#222222") + '"></i>' +
            '<span><strong>' + roles[index] + '</strong><em>' + escapeHtml(color.name || color.hex || "Color") + ' · ' + temperature + '</em></span>' +
            '<strong>' + weight + '%</strong>' +
          '</div>';
        }).join("");
      }

      function colorTemperatureLabel(rgb) {
        var red = Number(rgb[0] || 0);
        var green = Number(rgb[1] || 0);
        var blue = Number(rgb[2] || 0);
        var max = Math.max(red, green, blue) / 255;
        var min = Math.min(red, green, blue) / 255;
        var delta = max - min;
        var hue = 0;
        if (delta !== 0) {
          if (max === red / 255) hue = ((green - blue) / 255 / delta) % 6;
          else if (max === green / 255) hue = (blue - red) / 255 / delta + 2;
          else hue = (red - green) / 255 / delta + 4;
          hue = (hue * 60 + 360) % 360;
        }
        var saturation = max === 0 ? 0 : delta / max;
        if (saturation < 0.12) return "neutre";
        if (hue < 55 || hue >= 315) return "chaud";
        if (hue >= 55 && hue < 165) return "organique";
        if (hue >= 165 && hue < 260) return "froid";
        return "mystique";
      }

      function clearObjectUrls() {
        objectUrls.forEach(function(url) { URL.revokeObjectURL(url); });
        objectUrls = [];
      }

      function clearPreview() {
        if (previewAbort) previewAbort.abort();
        previewPages.innerHTML = "";
        previewPages.hidden = true;
        emptyPreview.textContent = "No pages generated.";
        emptyPreview.hidden = false;
      }

      function renderThumbs() {
        clearObjectUrls();
        thumbGrid.innerHTML = "";
        thumbGridNav.innerHTML = "";
        files.forEach(function(file, index) {
          var url = URL.createObjectURL(file);
          objectUrls.push(url);
          thumbGrid.appendChild(createThumb(file, index, url));
          thumbGridNav.appendChild(createThumb(file, index, url));
        });
      }

      function createThumb(file, index, url) {
        var item = document.createElement("div");
        item.className = "thumb";
        var img = document.createElement("img");
        img.alt = file.name;
        img.src = url;
        item.appendChild(img);
        var remove = document.createElement("button");
        remove.className = "thumb-remove";
        remove.type = "button";
        remove.setAttribute("aria-label", "Remove " + file.name);
        remove.innerHTML = '<span><svg aria-hidden="true"><use href="#icon-trash"></use></svg></span>';
        remove.addEventListener("click", function() {
          removeFileAt(index);
        });
        item.appendChild(remove);
        return item;
      }

      function setFiles(nextFiles) {
        var incoming = Array.prototype.filter.call(nextFiles, function(file) {
          return file.type.indexOf("image/") === 0 || /\.(jpe?g|png|webp|bmp|gif|tiff?)$/i.test(file.name);
        });
        files = files.concat(incoming);
        renderThumbs();
        updateMetrics();
        if (files.length) {
          setStatus("Images ready.");
          schedulePreview(120);
          scheduleAnalysis(320);
        } else {
          clearPreview();
          clearAnalysis();
          setStatus("Ready.");
        }
      }

      function removeFileAt(index) {
        files.splice(index, 1);
        renderThumbs();
        updateMetrics();
        if (files.length) {
          schedulePreview(120);
          scheduleAnalysis(320);
          setStatus("Image removed.");
        } else {
          clearPreview();
          clearAnalysis();
          setStatus("Ready.");
        }
      }

      function sortRects(rects) {
        rects.sort(function(a, b) {
          return a.y - b.y || a.x - b.x || (b.w * b.h) - (a.w * a.h);
        });
      }

      function splitRandomRect(rects, target) {
        while (rects.length < target) {
          var candidates = rects.filter(function(rect) { return rect.w > 1 || rect.h > 1; });
          if (!candidates.length) break;
          var rect = candidates[Math.floor(Math.random() * candidates.length)];
          rects.splice(rects.indexOf(rect), 1);
          var splitVertical = rect.w >= rect.h;
          if (rect.w > 1 && rect.h > 1) splitVertical = Math.random() > 0.35 ? rect.w >= rect.h : rect.w < rect.h;
          if (splitVertical && rect.w > 1) {
            var w1 = 1 + Math.floor(Math.random() * (rect.w - 1));
            rects.push({ x: rect.x, y: rect.y, w: w1, h: rect.h });
            rects.push({ x: rect.x + w1, y: rect.y, w: rect.w - w1, h: rect.h });
          } else if (rect.h > 1) {
            var h1 = 1 + Math.floor(Math.random() * (rect.h - 1));
            rects.push({ x: rect.x, y: rect.y, w: rect.w, h: h1 });
            rects.push({ x: rect.x, y: rect.y + h1, w: rect.w, h: rect.h - h1 });
          } else {
            rects.push(rect);
            break;
          }
        }
        sortRects(rects);
        return rects;
      }

      function randomizeCustomGrid() {
        var target = 8 + Math.floor(Math.random() * 7);
        customRects = splitRandomRect([{ x: 0, y: 0, w: 6, h: 4 }], target);
        customCells = rectsToCells(customRects);
        syncRectsFromCells();
        renderCustomGrid();
        schedulePreview(120);
      }

      function resetCustomGrid() {
        customCells = rectsToCells(defaultCustomRects);
        syncRectsFromCells();
        renderCustomGrid();
        schedulePreview(120);
      }

      function renderCustomGrid() {
        syncRectsFromCells();
        customGrid.innerHTML = "";
        for (var index = 0; index < 24; index += 1) {
          var button = document.createElement("button");
          var colorIndex = customCells[index] % customPalette.length;
          button.className = "custom-cell";
          button.style.setProperty("--cell-color", customPalette[colorIndex]);
          button.type = "button";
          button.setAttribute("aria-label", "Cell " + (index + 1));
          button.addEventListener("click", function(cellIndex) {
            return function() {
              customCells[cellIndex] = (customCells[cellIndex] + 1) % customPalette.length;
              syncRectsFromCells();
              renderCustomGrid();
              schedulePreview(120);
            };
          }(index));
          customGrid.appendChild(button);
        }
        document.getElementById("customGridCount").textContent = customSlotCount() + " blocks";
        updateMetrics();
      }

      function updateLayoutMode() {
        var isCustom = document.getElementById("layoutMode").value === "custom";
        customGridPanel.classList.toggle("is-visible", isCustom);
        updateMetrics();
        schedulePreview(180);
      }

      function customGridPayload() {
        return {
          cols: 6,
          rows: 4,
          rects: customRects.map(function(rect) {
            return { x: rect.x, y: rect.y, w: rect.w, h: rect.h };
          })
        };
      }

      function collectParams() {
        var formats = Array.prototype.map.call(document.querySelectorAll('input[name="format"]:checked'), function(input) {
          return input.value;
        });
        if (!formats.length) {
          document.getElementById("formatPdf").checked = true;
          formats = ["pdf"];
        }
        return {
          layoutMode: document.getElementById("layoutMode").value,
          imagesPerPage: Number(document.getElementById("imagesPerPage").value),
          bentoOptimizer: document.getElementById("bentoOptimizer").checked,
          bentoOptimizerMode: document.getElementById("bentoOptimizerMode").value,
          autoImagesPerPage: false,
          pagePreset: document.getElementById("pagePreset").value,
          pageWidth: Number(document.getElementById("pageWidth").value),
          pageHeight: Number(document.getElementById("pageHeight").value),
          margin: Number(document.getElementById("margin").value),
          gap: Number(document.getElementById("gap").value),
          borderRadius: Number(document.getElementById("borderRadius").value),
          fillBlocks: document.getElementById("fillBlocks").checked,
          backgroundColor: document.getElementById("backgroundText").value,
          useColorGradient: document.getElementById("useColorGradient").checked,
          gradientMode: document.getElementById("gradientMode").value,
          orientationThreshold: Number(document.getElementById("orientationThreshold").value),
          analysisDepth: document.getElementById("analysisDepth").value,
          customGrid: customGridPayload(),
          formats: formats
        };
      }

      function buildFormData() {
        var data = new FormData();
        data.append("params", JSON.stringify(collectParams()));
        files.forEach(function(file) {
          data.append("images", file, file.name);
        });
        return data;
      }

      function setActiveView(view) {
        activeView = view;
        var isGraph = view === "graph";
        document.body.classList.toggle("is-graph-view", isGraph);
        document.getElementById("bentoView").hidden = isGraph;
        document.getElementById("graphView").hidden = !isGraph;
        document.getElementById("bentoViewButton").classList.toggle("is-active", !isGraph);
        document.getElementById("graphViewButton").classList.toggle("is-active", isGraph);
        document.getElementById("bentoViewButton").setAttribute("aria-pressed", String(!isGraph));
        document.getElementById("graphViewButton").setAttribute("aria-pressed", String(isGraph));
        if (isGraph) {
          if (files.length && !analysisPayload) requestAnalysis();
          setTimeout(resizeGraphCanvas, 40);
          setTimeout(resizeHarmonyCanvas, 40);
          startGraphAnimation();
        } else {
          stopGraphAnimation();
        }
      }

      function clearAnalysis() {
        if (analysisAbort) analysisAbort.abort();
        if (analysisTimer) clearTimeout(analysisTimer);
        analysisPayload = null;
        analysisTagGroups = [];
        document.getElementById("analysisStatus").textContent = "Analysis idle.";
        document.getElementById("analysisList").innerHTML = "";
        document.getElementById("analysisTags").innerHTML = "";
        document.getElementById("graphDetail").innerHTML = '<span>Selected node</span><strong>No node selected</strong><p>Click a node in the graph to inspect its cluster, weight and connected relations.</p>';
        graphEmpty.textContent = "No graph generated.";
        graphEmpty.hidden = false;
        closeAnalysisTagModal();
        updateGraphMetrics();
        updateProfilePanel();
        drawGraph();
        drawHarmony();
      }

      function setAnalysisPending(message) {
        analysisPayload = null;
        document.getElementById("analysisStatus").textContent = message || "Analysis queued.";
        document.getElementById("analysisList").innerHTML = Array.from({ length: 4 }, function() {
          return '<div class="analysis-row"><i class="analysis-dot skeleton-chip"></i><span><b class="skeleton-line"></b></span><strong><b class="skeleton-line short"></b></strong></div>';
        }).join("");
        document.getElementById("analysisTags").innerHTML = '<div class="analysis-chip-row">' + Array.from({ length: 5 }, function() {
          return '<span class="analysis-tag-skeleton skeleton-chip"></span>';
        }).join("") + '</div>';
        document.getElementById("graphDetail").innerHTML = '<span>Selected node</span><strong>Waiting for graph</strong><p>The latent graph will appear when analysis is complete.</p>';
        document.getElementById("graphSubtitle").textContent = "Spectral analysis pending.";
        graphEmpty.textContent = "Building latent graph...";
        graphEmpty.hidden = false;
        updateGraphMetrics({ nodes: [], edges: [] });
        updateProfilePanel();
        drawGraph();
        drawHarmony();
      }

      function scheduleAnalysis(delay) {
        if (analysisTimer) clearTimeout(analysisTimer);
        if (!files.length) {
          clearAnalysis();
          return;
        }
        setAnalysisPending("Analysis queued.");
        analysisTimer = setTimeout(requestAnalysis, delay || 900);
      }

      async function requestAnalysis() {
        if (!files.length) return;
        analysisGeneration += 1;
        var generation = analysisGeneration;
        if (analysisAbort) analysisAbort.abort();
        analysisAbort = new AbortController();
        setAnalysisPending("Analyzing moodboard...");
        document.getElementById("analysisStatus").textContent = "Analyzing moodboard...";
        document.getElementById("graphSubtitle").textContent = "Analyzing...";
        graphEmpty.textContent = "Analyzing graph...";
        graphEmpty.hidden = false;
        try {
          var response = await fetch("/api/analyze", { method: "POST", body: buildFormData(), signal: analysisAbort.signal });
          if (!response.ok) {
            var errorPayload = await response.json().catch(function() { return {}; });
            throw new Error(errorPayload.error || "Analysis error.");
          }
          var payload = await response.json();
          if (generation !== analysisGeneration) return;
          analysisPayload = payload;
          renderAnalysis(payload);
          document.getElementById("analysisStatus").textContent = payload.cache && payload.cache.hit ? "Analysis cache hit." : "Analysis ready.";
          updateMetrics();
          resizeGraphCanvas();
          resizeHarmonyCanvas();
          drawGraph();
          drawHarmony();
        } catch (error) {
          if (error.name === "AbortError") return;
          document.getElementById("analysisStatus").textContent = error.message;
          document.getElementById("graphSubtitle").textContent = error.message;
          graphEmpty.textContent = error.message;
          graphEmpty.hidden = false;
        }
      }

      function renderAnalysis(payload) {
        var scores = payload.scores || {};
        var graph = payload.graph || { nodes: [], edges: [] };
        var spectral = payload.spectralAnalysis || {};
        var dominant = payload.globalProfile && payload.globalProfile.dominantAesthetic ? payload.globalProfile.dominantAesthetic : {};
        var harmonyScore = firstMetricValue(scores.harmonicity, scores.harmonyCoherence, spectral.harmonicityScore);
        prepareGraphLayout(graph);
        updateProfilePanel();
        document.getElementById("graphSubtitle").textContent = (dominant.name || "Unknown") + " - purity " + formatPercent(scores.purity) + " - spectral harmonicity " + formatPercent(harmonyScore);
        var clusters = payload.clusters || [];
        var clusterSelect = document.getElementById("graphClusterFilter");
        clusterSelect.innerHTML = '<option value="all">All</option>';
        clusters.forEach(function(cluster) {
          var option = document.createElement("option");
          option.value = String(cluster.id);
          option.textContent = cluster.label + " - " + cluster.size;
          clusterSelect.appendChild(option);
        });
        var list = document.getElementById("analysisList");
        var rows = [];
        rows.push(rowHtml("#5D71FC", "Dominant", dominant.name || "Unknown", formatPercent(dominant.score)));
        rows.push(rowHtml("#f89540", "Purity", "Moodboard purity", formatPercent(scores.purity)));
        rows.push(rowHtml("#5D71FC", "Spectral", "Laplacian harmonicity", formatPercent(spectral.harmonicityScore)));
        rows.push(rowHtml("#27AE60", "Palette", (payload.palette || []).slice(0, 3).map(function(color) { return color.name; }).join(", "), ""));
        rows.push(rowHtml("#f89540", "Harmony", "Coherence", formatPercent(harmonyScore)));
        rows.push(rowHtml("#27AE60", "Color", "Coherence", formatPercent(scores.colorCoherence)));
        (payload.aestheticMatches || []).slice(1, 5).forEach(function(match, index) {
          rows.push(rowHtml(graphColors[(index + 2) % graphColors.length], "Aesthetic", match.name, formatPercent(match.score)));
        });
        list.innerHTML = rows.join("");
        renderAnalysisTags(payload);
        document.getElementById("graphDetail").innerHTML = '<span>Selected node</span><strong>No node selected</strong><p>Click a node in the graph to inspect its cluster, weight and connected relations.</p>';
        updateGraphMetrics(graph);
      }

      function rowHtml(color, role, label, value) {
        return '<div class="analysis-row"><i class="analysis-dot" style="--dot:' + color + '"></i><span>' + escapeHtml(role + " - " + label) + '</span><strong>' + escapeHtml(value) + '</strong></div>';
      }

      function firstMetricValue() {
        for (var i = 0; i < arguments.length; i += 1) {
          var raw = arguments[i];
          if (raw === undefined || raw === null || raw === "") continue;
          var value = Number(raw);
          if (Number.isFinite(value)) return value;
        }
        return null;
      }

      function formatPercent() {
        var value = firstMetricValue.apply(null, arguments);
        return value === null ? "--" : Math.round(value * 100) + "%";
      }

      function graphDegreeMap(graph) {
        var stats = {};
        (graph.nodes || []).forEach(function(node) {
          stats[node.id] = { count: 0, weight: 0 };
        });
        (graph.edges || []).forEach(function(edge) {
          var weight = Number(edge.weight) || 0;
          if (stats[edge.source]) {
            stats[edge.source].count += 1;
            stats[edge.source].weight += weight;
          }
          if (stats[edge.target]) {
            stats[edge.target].count += 1;
            stats[edge.target].weight += weight;
          }
        });
        return stats;
      }

      function analysisCategoryForNode(node) {
        if (node.type === "aesthetic") return "aesthetics";
        if (node.type === "color") return "colors";
        if (node.type === "object") return "objects";
        if (node.type === "emotion") return "emotions";
        if (node.type === "affect") return "affects";
        if (node.type === "texture") return "textures";
        if (node.type === "style") return "styles";
        if (node.type === "composition") return "composition";
        if (node.type === "symbol") return "symbols";
        return "";
      }

      function analysisCategoryMeta(key) {
        return {
          aesthetics: { label: "Esthetique", color: "#EB5757" },
          objects: { label: "Objets", color: "#f89540" },
          symbols: { label: "Symboles", color: "#A855F7" },
          emotions: { label: "Emotions", color: "#5D71FC" },
          affects: { label: "Valeurs", color: "#EB5757" },
          textures: { label: "Textures", color: "#C58CA8" },
          styles: { label: "Styles", color: "#27AE60" },
          composition: { label: "Composition", color: "#56CCF2" },
          colors: { label: "Couleurs", color: "#F2C94C" }
        }[key] || { label: key, color: "#5D71FC" };
      }

      function buildAnalysisTagGroups(payload) {
        var graph = payload && payload.graph ? payload.graph : { nodes: [], edges: [] };
        var degree = graphDegreeMap(graph);
        var groups = {};
        (graph.nodes || []).forEach(function(node) {
          var key = analysisCategoryForNode(node);
          if (!key) return;
          var stat = degree[node.id] || { count: 0, weight: 0 };
          var meta = analysisCategoryMeta(key);
          if (!groups[key]) groups[key] = { key: key, label: meta.label, color: meta.color, items: [] };
          groups[key].items.push({
            id: node.id,
            label: node.label || node.id,
            color: node.clusterColor || meta.color,
            cluster: node.cluster,
            count: stat.count,
            weight: stat.weight,
            type: node.type
          });
        });
        return ["aesthetics", "objects", "symbols", "affects", "textures", "emotions", "styles", "composition", "colors"].map(function(key) {
          var group = groups[key];
          if (!group || !group.items.length) return null;
          group.items.sort(function(left, right) {
            if (right.count !== left.count) return right.count - left.count;
            if (right.weight !== left.weight) return right.weight - left.weight;
            return String(left.label).localeCompare(String(right.label));
          });
          return group;
        }).filter(Boolean);
      }

      function tagChipHtml(item, groupKey, more) {
        if (more) {
          return '<button class="analysis-tag-chip is-more" type="button" data-tag-more="' + escapeHtml(groupKey) + '" aria-label="Show all ' + escapeHtml(groupKey) + ' tags">&hellip;</button>';
        }
        return '<button class="analysis-tag-chip" type="button" data-tag-node="' + escapeHtml(item.id) + '" title="' + escapeHtml(item.count + " graph connections") + '" style="--tag-color:' + escapeHtml(item.color) + '">' + escapeHtml(item.label) + '</button>';
      }

      function renderAnalysisTags(payload) {
        var container = document.getElementById("analysisTags");
        analysisTagGroups = buildAnalysisTagGroups(payload);
        if (!analysisTagGroups.length) {
          container.innerHTML = "";
          return;
        }
        container.innerHTML = analysisTagGroups.map(function(group) {
          var visibleItems = group.items.slice(0, 5).map(function(item) {
            return tagChipHtml(item, group.key, false);
          });
          if (group.items.length > 5) visibleItems.push(tagChipHtml(null, group.key, true));
          return '<section class="analysis-tag-group"><div class="analysis-tag-heading"><span>' + escapeHtml(group.label) + '</span><strong>' + group.items.length + '</strong></div><div class="analysis-chip-row">' + visibleItems.join("") + '</div></section>';
        }).join("");
      }

      function findAnalysisTagGroup(key) {
        return analysisTagGroups.find(function(group) { return group.key === key; }) || null;
      }

      function openAnalysisTagModal(key) {
        var group = findAnalysisTagGroup(key);
        if (!group) return;
        document.getElementById("analysisTagModalTitle").textContent = group.label;
        document.getElementById("analysisTagModalMeta").textContent = group.items.length + " detected tags";
        document.getElementById("analysisTagModalList").innerHTML = group.items.map(function(item) {
          return tagChipHtml(item, group.key, false);
        }).join("");
        document.getElementById("analysisTagModal").hidden = false;
      }

      function closeAnalysisTagModal() {
        var modal = document.getElementById("analysisTagModal");
        if (modal) modal.hidden = true;
      }

      function selectGraphNode(nodeId) {
        if (!analysisPayload || !analysisPayload.graph) return;
        var node = (analysisPayload.graph.nodes || []).find(function(item) { return item.id === nodeId; });
        if (!node) return;
        graphState.selectedNodeId = node.id;
        renderGraphDetail(node);
        drawGraph();
      }

      function handleAnalysisTagClick(event) {
        if (!event.target || !event.target.closest) return;
        var more = event.target.closest("[data-tag-more]");
        if (more) {
          openAnalysisTagModal(more.getAttribute("data-tag-more"));
          return;
        }
        var chip = event.target.closest("[data-tag-node]");
        if (chip) {
          selectGraphNode(chip.getAttribute("data-tag-node"));
        }
      }

      function escapeHtml(value) {
        return String(value || "").replace(/[&<>"']/g, function(ch) {
          return {"&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#039;"}[ch];
        });
      }

function schedulePreview(delay) {
        updateMetrics();
        if (previewTimer) clearTimeout(previewTimer);
        if (!files.length) return;
        previewTimer = setTimeout(requestPreview, delay || 650);
      }

      async function requestPreview() {
        if (!files.length) return;
        previewGeneration += 1;
        var generation = previewGeneration;
        if (previewAbort) previewAbort.abort();
        previewAbort = new AbortController();
        setStatus("Rendering preview...");
        emptyPreview.textContent = "Generating preview...";
        emptyPreview.hidden = false;
        previewPages.hidden = true;
        try {
          var response = await fetch("/api/preview", { method: "POST", body: buildFormData(), signal: previewAbort.signal });
          if (!response.ok) {
            var errorPayload = await response.json().catch(function() { return {}; });
            throw new Error(errorPayload.error || "Preview error.");
          }
          var payload = await response.json();
          if (generation !== previewGeneration) return;
          if (!payload.pages || !payload.pages.length) {
            throw new Error("No preview page received.");
          }
          previewPages.innerHTML = "";
          payload.pages.forEach(function(src, index) {
            var page = document.createElement("figure");
            page.className = "preview-page";
            var image = document.createElement("img");
            // Local previews use cached URLs; serverless previews may return inline data URLs.
            image.src = src;
            image.alt = "Moodboard page " + (index + 1);
            image.addEventListener("error", function() {
              setStatus("A preview page could not load.");
            });
            page.appendChild(image);
            previewPages.appendChild(page);
          });
          previewPages.hidden = false;
          emptyPreview.hidden = true;
          updateMetrics(Number(payload.pageCount || 0), Number(payload.imagesPerPage || 0));
          setStatus("PNG preview.");
        } catch (error) {
          if (error.name === "AbortError") return;
          emptyPreview.textContent = error.message;
          emptyPreview.hidden = false;
          previewPages.hidden = true;
          setStatus(error.message);
        }
      }

      function ensureFiles() {
        if (!files.length) {
          setStatus("Add at least one image.");
          return false;
        }
        return true;
      }

      function filenameFromDisposition(disposition) {
        if (!disposition) return "moodboard_bento_export";
        var match = /filename="([^"]+)"/.exec(disposition);
        return match ? match[1] : "moodboard_bento_export";
      }

      async function requestExport() {
        if (!ensureFiles()) return;
        setStatus("Generating export...");
        generateButton.disabled = true;
        try {
          var response = await fetch("/api/generate", { method: "POST", body: buildFormData() });
          if (!response.ok) {
            var errorPayload = await response.json().catch(function() { return {}; });
            throw new Error(errorPayload.error || "Generation error.");
          }
          var blob = await response.blob();
          var filename = filenameFromDisposition(response.headers.get("Content-Disposition"));
          var url = URL.createObjectURL(blob);
          var link = document.createElement("a");
          link.href = url;
          link.download = filename;
          document.body.appendChild(link);
          link.click();
          link.remove();
          setTimeout(function() { URL.revokeObjectURL(url); }, 2000);
          var pages = Number(response.headers.get("X-Page-Count") || 0);
          var perPage = Number(response.headers.get("X-Images-Per-Page") || 0);
          updateMetrics(pages, perPage);
          setStatus("Document downloaded.");
        } catch (error) {
          setStatus(error.message);
        } finally {
          generateButton.disabled = false;
        }
      }

      function buildMiniGrids() {
        document.querySelectorAll(".mini-grid").forEach(function(grid) {
          for (var i = 0; i < 24; i += 1) {
            grid.appendChild(document.createElement("i"));
          }
        });
      }

      function updateThemeIcon() {
        var isDark = document.body.dataset.theme === "dark";
        document.getElementById("themeIcon").setAttribute("href", isDark ? "#icon-sun" : "#icon-moon");
      }

      async function setPreviewFullscreen(enabled, options) {
        options = options || {};
        if (enabled && document.activeElement instanceof HTMLElement) document.activeElement.blur();
        document.body.classList.toggle("is-fullscreen-preview", enabled);
        fullscreenPreview.classList.toggle("is-active", enabled);
        fullscreenPreview.setAttribute("aria-pressed", String(enabled));
        fullscreenPreview.setAttribute("aria-label", enabled ? "Exit fullscreen" : "Open preview fullscreen");
        fullscreenPreview.title = enabled ? "Exit fullscreen (q)" : "Fullscreen";
        document.getElementById("fullscreenIcon").setAttribute("href", enabled ? "#icon-minimize" : "#icon-fullscreen");

        if (!options.skipNative) {
          try {
            if (enabled && document.fullscreenEnabled && !document.fullscreenElement) {
              await document.documentElement.requestFullscreen();
            } else if (!enabled && document.fullscreenElement) {
              await document.exitFullscreen();
            }
          } catch (_error) {
            // App-level fullscreen remains available.
          }
        }
      }

      async function setGraphFullscreen(enabled, options) {
        options = options || {};
        if (enabled && document.activeElement instanceof HTMLElement) document.activeElement.blur();
        if (enabled) setActiveView("graph");
        document.body.classList.toggle("is-fullscreen-graph", enabled);
        var button = document.getElementById("fullscreenGraph");
        button.classList.toggle("is-active", enabled);
        button.setAttribute("aria-pressed", String(enabled));
        button.setAttribute("aria-label", enabled ? "Exit graph fullscreen" : "Open graph fullscreen");
        button.title = enabled ? "Exit fullscreen (q)" : "Fullscreen";
        document.getElementById("fullscreenGraphIcon").setAttribute("href", enabled ? "#icon-minimize" : "#icon-fullscreen");
        setTimeout(resizeGraphCanvas, 40);
        setTimeout(resizeHarmonyCanvas, 40);

        if (!options.skipNative) {
          try {
            if (enabled && document.fullscreenEnabled && !document.fullscreenElement) {
              await document.documentElement.requestFullscreen();
            } else if (!enabled && document.fullscreenElement) {
              await document.exitFullscreen();
            }
          } catch (_error) {
            // App-level fullscreen remains available.
          }
        }
      }

      imageInput.addEventListener("change", function(event) {
        setFiles(event.target.files || []);
        imageInput.value = "";
      });

      clearButton.addEventListener("click", function() {
        imageInput.value = "";
        files = [];
        setFiles([]);
      });

      ["dragenter", "dragover"].forEach(function(type) {
        dropZone.addEventListener(type, function(event) {
          event.preventDefault();
          dropZone.classList.add("is-over");
        });
      });

      ["dragleave", "drop"].forEach(function(type) {
        dropZone.addEventListener(type, function(event) {
          event.preventDefault();
          dropZone.classList.remove("is-over");
        });
      });

      dropZone.addEventListener("drop", function(event) {
        setFiles(event.dataTransfer.files || []);
      });

      document.getElementById("bentoViewButton").addEventListener("click", function() {
        setActiveView("bento");
      });
      document.getElementById("graphViewButton").addEventListener("click", function() {
        setActiveView("graph");
      });
      generateButton.addEventListener("click", requestExport);
      fullscreenPreview.addEventListener("click", function() {
        setPreviewFullscreen(!document.body.classList.contains("is-fullscreen-preview"));
      });
      document.getElementById("fullscreenGraph").addEventListener("click", function() {
        setGraphFullscreen(!document.body.classList.contains("is-fullscreen-graph"));
      });
      document.getElementById("randomCustomGrid").addEventListener("click", randomizeCustomGrid);
      document.getElementById("resetCustomGrid").addEventListener("click", resetCustomGrid);

      document.querySelectorAll('input[type="range"]').forEach(function(input) {
        updateRange(input);
        input.addEventListener("input", function() {
          if (input.id === "graphEdgeFilter") return;
          if (input.id === "imagesPerPage") manualImagesPerPage = Number(input.value) || manualImagesPerPage;
          updateRange(input);
          schedulePreview();
          if (input.id === "orientationThreshold") scheduleAnalysis();
        });
      });

      document.getElementById("pagePreset").addEventListener("change", function() {
        updatePresetFields();
        schedulePreview(180);
      });
      document.getElementById("bentoOptimizer").addEventListener("change", function() {
        updateOptimizerState();
        schedulePreview(180);
      });
      document.getElementById("bentoOptimizerMode").addEventListener("change", function() {
        updateMetrics();
        schedulePreview(180);
      });
      document.getElementById("layoutMode").addEventListener("change", updateLayoutMode);
      document.getElementById("backgroundColor").addEventListener("input", function() {
        updateBackgroundFromColor();
        schedulePreview();
      });
      document.getElementById("backgroundText").addEventListener("input", function() {
        updateBackgroundFromText();
        schedulePreview();
      });

      ["fillBlocks", "useColorGradient", "gradientMode", "pageWidth", "pageHeight"].forEach(function(id) {
        document.getElementById(id).addEventListener("change", function() {
          schedulePreview(180);
        });
      });

      document.getElementById("graphTypeFilter").addEventListener("change", function() {
        graphState.typeFilter = this.value;
        drawGraph();
      });
      document.getElementById("graphClusterFilter").addEventListener("change", function() {
        graphState.clusterFilter = this.value;
        drawGraph();
      });
      document.getElementById("graphEdgeFilter").addEventListener("input", function() {
        graphState.minEdge = Number(this.value);
        updateRange(this, Number(this.value).toFixed(2));
        drawGraph();
      });
      document.getElementById("graphLabelToggle").addEventListener("change", function() {
        graphState.showLabels = this.checked;
        drawGraph();
      });
      document.getElementById("resetGraphView").addEventListener("click", function() {
        graphState.zoom = 1;
        graphState.panX = 0;
        graphState.panY = 0;
        drawGraph();
      });
      document.getElementById("exportGraphPng").addEventListener("click", exportGraphPng);
      document.getElementById("exportGraphJson").addEventListener("click", exportGraphJson);
      document.getElementById("analysisTags").addEventListener("click", handleAnalysisTagClick);
      document.getElementById("analysisTagModalList").addEventListener("click", handleAnalysisTagClick);
      document.getElementById("analysisTagModalClose").addEventListener("click", closeAnalysisTagModal);
      document.getElementById("analysisTagModal").addEventListener("click", function(event) {
        if (event.target === this) closeAnalysisTagModal();
      });

      graphCanvas.addEventListener("pointerdown", function(event) {
        var node = pickGraphNode(event.offsetX, event.offsetY);
        graphState.dragging = true;
        graphState.draggingNode = node ? node.id : null;
        graphState.selectedNodeId = node ? node.id : graphState.selectedNodeId;
        if (node) renderGraphDetail(node);
        graphState.dragStartX = event.clientX;
        graphState.dragStartY = event.clientY;
        graphState.panStartX = graphState.panX;
        graphState.panStartY = graphState.panY;
        graphCanvas.setPointerCapture(event.pointerId);
        drawGraph();
      });

      graphCanvas.addEventListener("pointermove", function(event) {
        if (graphState.dragging) {
          var rect = graphCanvas.getBoundingClientRect();
          if (graphState.draggingNode && analysisPayload && analysisPayload.graph) {
            var world = inverseProjectGraph(event.offsetX, event.offsetY, rect.width, rect.height);
            analysisPayload.graph.nodes.forEach(function(node) {
              if (node.id === graphState.draggingNode) {
                node.x = world.x;
                node.y = world.y;
                node.manualPosition = true;
              }
            });
          } else {
            graphState.panX = graphState.panStartX + (event.clientX - graphState.dragStartX);
            graphState.panY = graphState.panStartY + (event.clientY - graphState.dragStartY);
          }
          drawGraph();
          return;
        }
        var node = pickGraphNode(event.offsetX, event.offsetY);
        graphState.hoverNodeId = node ? node.id : null;
        updateGraphTooltip(event, node);
        drawGraph();
      });

      graphCanvas.addEventListener("pointerup", function(event) {
        graphState.dragging = false;
        graphState.draggingNode = null;
        try { graphCanvas.releasePointerCapture(event.pointerId); } catch (_error) {}
      });

      graphCanvas.addEventListener("pointerleave", function() {
        graphState.hoverNodeId = null;
        graphState.dragging = false;
        graphState.draggingNode = null;
        graphTooltip.style.display = "none";
        graphTooltip.classList.remove("is-image-preview");
        drawGraph();
      });

      graphCanvas.addEventListener("wheel", function(event) {
        event.preventDefault();
        graphState.zoom = Math.max(0.45, Math.min(4.5, graphState.zoom * (event.deltaY > 0 ? 0.9 : 1.12)));
        drawGraph();
      }, { passive: false });

      window.addEventListener("resize", function() {
        resizeGraphCanvas();
        resizeHarmonyCanvas();
      });

      document.getElementById("themeToggle").addEventListener("click", function() {
        var next = document.body.dataset.theme === "dark" ? "light" : "dark";
        document.body.dataset.theme = next;
        this.setAttribute("aria-pressed", String(next === "dark"));
        updateThemeIcon();
        drawGraph();
      });

      document.addEventListener("keydown", function(event) {
        if (event.key === "Escape" && !document.getElementById("analysisTagModal").hidden) {
          event.preventDefault();
          closeAnalysisTagModal();
          return;
        }
        var target = event.target;
        var typing = target instanceof HTMLInputElement || target instanceof HTMLSelectElement || target instanceof HTMLTextAreaElement;
        if (typing) return;
        if (event.key.toLowerCase() === "q" && document.body.classList.contains("is-fullscreen-preview")) {
          event.preventDefault();
          setPreviewFullscreen(false);
        }
        if (event.key.toLowerCase() === "q" && document.body.classList.contains("is-fullscreen-graph")) {
          event.preventDefault();
          setGraphFullscreen(false);
        }
      });

      document.addEventListener("fullscreenchange", function() {
        if (!document.fullscreenElement && document.body.classList.contains("is-fullscreen-preview")) {
          setPreviewFullscreen(false, { skipNative: true });
        }
        if (!document.fullscreenElement && document.body.classList.contains("is-fullscreen-graph")) {
          setGraphFullscreen(false, { skipNative: true });
        }
      });

      document.querySelectorAll('input[name="format"]').forEach(function(input) {
        input.addEventListener("change", updateMetrics);
      });

      customCells = rectsToCells(defaultCustomRects);
      syncRectsFromCells();
      buildMiniGrids();
      renderCustomGrid();
      updatePresetFields();
      updateOptimizerState();
      updateLayoutMode();
      updateThemeIcon();
      updateMetrics();
      resizeGraphCanvas();
      resizeHarmonyCanvas();
      drawGraph();
      drawHarmony();
