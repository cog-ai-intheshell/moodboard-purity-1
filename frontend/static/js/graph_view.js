// Graph rendering and interaction helpers for the Moodboard graph view.
// These functions intentionally use the shared globals initialized by bento_view.js.

      function updateGraphMetrics(graph) {
        graph = graph || (analysisPayload && analysisPayload.graph) || { nodes: [], edges: [] };
        var clusterIds = {};
        (graph.nodes || []).forEach(function(node) { clusterIds[node.cluster] = true; });
        document.getElementById("metricGraphNodes").textContent = String((graph.nodes || []).length);
        document.getElementById("metricGraphEdges").textContent = String((graph.edges || []).length);
        document.getElementById("metricGraphClusters").textContent = String(Object.keys(clusterIds).length);
        document.getElementById("metricGraphPurity").textContent = analysisPayload && analysisPayload.scores ? Math.round((analysisPayload.scores.purity || 0) * 100) + "%" : "--";
      }

      function cssVar(name, fallback) {
        return getComputedStyle(document.body).getPropertyValue(name).trim() || fallback;
      }

      function hexToRgb(hex) {
        var clean = String(hex || "#5D71FC").replace("#", "");
        if (clean.length !== 6) return [93, 113, 252];
        return [parseInt(clean.slice(0, 2), 16), parseInt(clean.slice(2, 4), 16), parseInt(clean.slice(4, 6), 16)];
      }

      function colorToRgba(color, alpha) {
        var text = String(color || "#5D71FC").trim();
        if (text.indexOf("rgb") === 0) {
          return text.replace(/rgba?\(([^)]+)\)/, function(_match, values) {
            var parts = values.split(",").slice(0, 3).join(",");
            return "rgba(" + parts + "," + alpha + ")";
          });
        }
        var rgb = hexToRgb(text);
        return "rgba(" + rgb[0] + ", " + rgb[1] + ", " + rgb[2] + ", " + alpha + ")";
      }

      function mixHex(left, right, amount) {
        var a = hexToRgb(left);
        var b = hexToRgb(right);
        var t = Math.max(0, Math.min(1, amount));
        return "#" + a.map(function(value, index) {
          var mixed = Math.round(value + (b[index] - value) * t);
          return mixed.toString(16).padStart(2, "0");
        }).join("");
      }

      function graphColor(node) {
        if (node.clusterColor) return node.clusterColor;
        var graph = analysisPayload && analysisPayload.graph ? analysisPayload.graph : null;
        var colors = graph && graph.clusterColors ? graph.clusterColors : null;
        if (colors && colors[String(node.cluster)]) return colors[String(node.cluster)];
        return graphColors[Math.abs(Number(node.cluster) || 0) % graphColors.length];
      }

      function graphSeed(value) {
        var text = String(value || "");
        var hash = 0;
        for (var i = 0; i < text.length; i += 1) {
          hash = ((hash << 5) - hash + text.charCodeAt(i)) | 0;
        }
        return Math.abs(hash);
      }

      function graphMotion(node) {
        if (!graphState.motionEnabled || graphState.draggingNode === node.id) return { x: 0, y: 0 };
        var seed = graphSeed(node.id);
        var time = graphState.motionTime * 0.001;
        var weight = Math.max(0.2, Math.min(1.0, Number(node.weight) || 0.4));
        var amplitude = (node.type === "image" ? 4.8 : 6.4) * (0.75 + weight * 0.55);
        return {
          x: Math.cos(time * (0.58 + (seed % 7) * 0.045) + seed * 0.021) * amplitude,
          y: Math.sin(time * (0.66 + (seed % 11) * 0.035) + seed * 0.017) * amplitude
        };
      }

      function graphAtomBase(node) {
        if (!graphState.atomLayout || node.manualPosition) {
          return { x: Number(node.x) || 0, y: Number(node.y) || 0 };
        }
        var seed = graphSeed(node.id);
        var cluster = Math.abs(Number(node.cluster) || 0);
        var orbitByType = {
          aesthetic: 0.08,
          image: 0.42,
          color: 0.56,
          emotion: 0.72,
          symbol: 0.84,
          style: 0.94,
          composition: 1.02
        };
        var orbit = orbitByType[node.type] || 0.78;
        var jitter = ((seed % 19) / 18 - 0.5) * 0.16;
        var angle = (seed % 6283) / 1000 + cluster * 0.78 + graphState.motionTime * 0.00008 * (node.type === "aesthetic" ? 0.35 : 1);
        var backendX = Number(node.x) || 0;
        var backendY = Number(node.y) || 0;
        if (node.type === "aesthetic") {
          return {
            x: Math.cos(angle) * (orbit + jitter * 0.22) + backendX * 0.10,
            y: Math.sin(angle) * (orbit + jitter * 0.22) * 0.58 + backendY * 0.10
          };
        }
        return {
          x: Math.cos(angle) * (orbit + jitter) + backendX * 0.20,
          y: Math.sin(angle) * (orbit + jitter) * 0.68 + backendY * 0.18
        };
      }

      function normalizeProjectedGraph(graph) {
        var nodes = graph.nodes || [];
        if (!nodes.length) return;
        var xs = nodes.map(function(node) { return Number(node.x) || 0; });
        var ys = nodes.map(function(node) { return Number(node.y) || 0; });
        var meanX = xs.reduce(function(sum, value) { return sum + value; }, 0) / nodes.length;
        var meanY = ys.reduce(function(sum, value) { return sum + value; }, 0) / nodes.length;
        var maxRadius = 0;
        nodes.forEach(function(node) {
          var x = (Number(node.x) || 0) - meanX;
          var y = (Number(node.y) || 0) - meanY;
          maxRadius = Math.max(maxRadius, Math.hypot(x, y));
        });
        maxRadius = Math.max(maxRadius, 1e-6);
        nodes.forEach(function(node) {
          var baseX = ((Number(node.x) || 0) - meanX) / maxRadius;
          var baseY = ((Number(node.y) || 0) - meanY) / maxRadius;
          var radius = Math.hypot(baseX, baseY);
          if (radius > 1e-6) {
            var expanded = Math.pow(radius, nodes.length > 24 ? 0.82 : 0.90) * 1.10;
            baseX = (baseX / radius) * expanded;
            baseY = (baseY / radius) * expanded;
          }
          node.latentX = baseX;
          node.latentY = baseY;
          node.x = baseX;
          node.y = baseY;
        });
      }

      function spreadProjectedGraphLayout(graph) {
        var nodes = graph.nodes || [];
        if (nodes.length < 2) return;
        normalizeProjectedGraph(graph);
        var byId = {};
        var degree = {};
        nodes.forEach(function(node) {
          byId[node.id] = node;
          degree[node.id] = 0;
        });
        var edges = (graph.edges || []).filter(function(edge) {
          return byId[edge.source] && byId[edge.target];
        });
        edges.forEach(function(edge) {
          var weight = Math.max(0.04, Number(edge.weight) || 0.18);
          degree[edge.source] += weight;
          degree[edge.target] += weight;
        });
        var iterations = nodes.length > 120 ? 90 : 140;
        for (var iteration = 0; iteration < iterations; iteration += 1) {
          var deltas = {};
          nodes.forEach(function(node) {
            deltas[node.id] = { x: 0, y: 0 };
          });
          for (var i = 0; i < nodes.length; i += 1) {
            for (var j = i + 1; j < nodes.length; j += 1) {
              var left = nodes[i];
              var right = nodes[j];
              var dx = (Number(right.x) || 0) - (Number(left.x) || 0);
              var dy = (Number(right.y) || 0) - (Number(left.y) || 0);
              var distance = Math.max(0.002, Math.hypot(dx, dy));
              var leftImportant = left.type === "image" || left.type === "aesthetic";
              var rightImportant = right.type === "image" || right.type === "aesthetic";
              var desired = (leftImportant || rightImportant) ? 0.115 : 0.075;
              if (left.cluster === right.cluster) desired *= 0.84;
              if (distance < desired) {
                var force = (desired - distance) * (left.cluster === right.cluster ? 0.032 : 0.052);
                var fx = (dx / distance) * force;
                var fy = (dy / distance) * force;
                deltas[left.id].x -= fx;
                deltas[left.id].y -= fy;
                deltas[right.id].x += fx;
                deltas[right.id].y += fy;
              }
            }
          }
          edges.forEach(function(edge) {
            var source = byId[edge.source];
            var target = byId[edge.target];
            var weight = Math.max(0.04, Number(edge.weight) || 0.18);
            var dx = (Number(target.x) || 0) - (Number(source.x) || 0);
            var dy = (Number(target.y) || 0) - (Number(source.y) || 0);
            var distance = Math.max(0.002, Math.hypot(dx, dy));
            var ideal = 0.23 + (1 - Math.min(1, weight)) * 0.36;
            if (source.cluster !== target.cluster) ideal += 0.12;
            var pull = (distance - ideal) * 0.0065 * Math.min(1.15, weight + 0.35);
            deltas[source.id].x += (dx / distance) * pull;
            deltas[source.id].y += (dy / distance) * pull;
            deltas[target.id].x -= (dx / distance) * pull;
            deltas[target.id].y -= (dy / distance) * pull;
          });
          nodes.forEach(function(node) {
            var anchor = node.type === "color" ? 0.020 : 0.033;
            deltas[node.id].x += ((Number(node.latentX) || 0) - (Number(node.x) || 0)) * anchor;
            deltas[node.id].y += ((Number(node.latentY) || 0) - (Number(node.y) || 0)) * anchor;
          });
          var cooling = 0.86 * (1 - iteration / (iterations * 1.15));
          nodes.forEach(function(node) {
            var delta = deltas[node.id];
            var nextX = (Number(node.x) || 0) + Math.max(-0.045, Math.min(0.045, delta.x * cooling));
            var nextY = (Number(node.y) || 0) + Math.max(-0.045, Math.min(0.045, delta.y * cooling));
            var radius = Math.hypot(nextX, nextY);
            var maxRadius = node.type === "color" ? 1.34 : 1.26;
            if (radius > maxRadius) {
              nextX = (nextX / radius) * maxRadius;
              nextY = (nextY / radius) * maxRadius;
            }
            node.x = nextX;
            node.y = nextY;
          });
        }
      }

      function graphVisibleNodeSet(graph) {
        var visible = {};
        (graph.nodes || []).forEach(function(node) {
          var typeOk = graphState.typeFilter === "all" || node.type === graphState.typeFilter;
          var clusterOk = graphState.clusterFilter === "all" || String(node.cluster) === graphState.clusterFilter;
          if (typeOk && clusterOk) visible[node.id] = true;
        });
        return visible;
      }

      function graphEdgeKey(edge) {
        return [edge.source, edge.target, edge.type || "relation"].join("|");
      }

      function isGraphMembershipEdge(edge) {
        var type = String(edge.type || "");
        return [
          "aesthetic_match",
          "color_affinity",
          "co_occurrence",
          "composition_affinity",
          "emotion_affinity",
          "style_affinity",
          "texture_affinity"
        ].indexOf(type) !== -1;
      }

      function graphEssentialEdgeMap(graph, visible) {
        var required = {};
        var bestByNode = {};
        (graph.edges || []).forEach(function(edge) {
          if (!visible[edge.source] || !visible[edge.target]) return;
          var key = graphEdgeKey(edge);
          var weight = Number(edge.weight) || 0;
          if (isGraphMembershipEdge(edge)) required[key] = true;
          [edge.source, edge.target].forEach(function(nodeId) {
            if (!bestByNode[nodeId] || weight > bestByNode[nodeId].weight) {
              bestByNode[nodeId] = { key: key, weight: weight };
            }
          });
        });
        Object.keys(visible).forEach(function(nodeId) {
          if (bestByNode[nodeId]) required[bestByNode[nodeId].key] = true;
        });
        return required;
      }

      function graphEdgeVisible(edge, visible, essentialEdges) {
        if (!visible[edge.source] || !visible[edge.target]) return false;
        if (essentialEdges && essentialEdges[graphEdgeKey(edge)]) return true;
        return Number(edge.weight) >= graphState.minEdge;
      }

      function graphFocusSet(graph, visible, essentialEdges) {
        var focusId = graphState.hoverNodeId;
        if (!focusId || !visible[focusId]) return null;
        var focus = {};
        focus[focusId] = true;
        (graph.edges || []).forEach(function(edge) {
          if (!graphEdgeVisible(edge, visible, essentialEdges)) return;
          if (edge.source === focusId) focus[edge.target] = true;
          if (edge.target === focusId) focus[edge.source] = true;
        });
        return { id: focusId, nodes: focus };
      }

      function prepareGraphLayout(graph) {
        if (!graph || graph.layoutPrepared) return;
        var nodes = graph.nodes || [];
        if (nodes.length < 2) {
          graph.layoutPrepared = true;
          return;
        }
        if (graph.projection && graph.projection.method) {
          spreadProjectedGraphLayout(graph);
          graph.layoutPrepared = true;
          return;
        }
        var byId = {};
        var degree = {};
        var adjacency = {};
        nodes.forEach(function(node) {
          byId[node.id] = node;
          degree[node.id] = 0;
          adjacency[node.id] = [];
        });
        var edges = (graph.edges || []).filter(function(edge) {
          return byId[edge.source] && byId[edge.target];
        });
        edges.forEach(function(edge) {
          var weight = Math.max(0.05, Number(edge.weight) || 0.25);
          degree[edge.source] += weight;
          degree[edge.target] += weight;
          adjacency[edge.source].push({ id: edge.target, weight: weight });
          adjacency[edge.target].push({ id: edge.source, weight: weight });
        });

        var ranked = nodes.slice().sort(function(left, right) {
          return (degree[right.id] || 0) - (degree[left.id] || 0);
        });
        var hubTarget = Math.max(3, Math.min(10, Math.ceil(Math.sqrt(nodes.length) * 1.45)));
        var hubs = ranked.slice(0, hubTarget);
        var hubIds = {};
        var golden = Math.PI * (3 - Math.sqrt(5));
        hubs.forEach(function(hub, index) {
          hubIds[hub.id] = true;
          var radius = index === 0 ? 0.08 : 0.20 + 0.42 * Math.sqrt(index / Math.max(1, hubs.length - 1));
          var angle = index * golden + (graphSeed(hub.id) % 97) / 97 + Math.PI * 0.16;
          hub.x = Math.cos(angle) * radius;
          hub.y = Math.sin(angle) * radius;
        });

        var satelliteCounts = {};
        hubs.forEach(function(hub) { satelliteCounts[hub.id] = 0; });
        var outerIndex = 0;
        nodes.forEach(function(node, index) {
          if (hubIds[node.id]) return;
          var neighbors = (adjacency[node.id] || []).slice().sort(function(left, right) {
            var leftHub = hubIds[left.id] ? 2 : 0;
            var rightHub = hubIds[right.id] ? 2 : 0;
            return (rightHub + right.weight + (degree[right.id] || 0) * 0.04) - (leftHub + left.weight + (degree[left.id] || 0) * 0.04);
          });
          var primary = neighbors.find(function(item) { return hubIds[item.id]; }) || neighbors[0];
          var seed = graphSeed(node.id);
          var lowDegree = (degree[node.id] || 0) < 0.72;
          if (!primary || lowDegree) {
            var outerAngle = outerIndex * golden + (seed % 360) * Math.PI / 180;
            var outerRadius = 0.72 + ((outerIndex % 5) * 0.062) + ((seed % 23) / 23) * 0.045;
            outerRadius = Math.min(0.98, outerRadius);
            node.x = Math.cos(outerAngle) * outerRadius;
            node.y = Math.sin(outerAngle) * outerRadius;
            outerIndex += 1;
            return;
          }
          var hub = byId[primary.id] || hubs[index % hubs.length];
          var count = satelliteCounts[hub.id] || 0;
          satelliteCounts[hub.id] = count + 1;
          var localAngle = count * golden + (seed % 157) / 157;
          var localRadius = 0.13 + (count % 5) * 0.038 + (node.type === "image" ? 0.075 : 0.018);
          node.x = (Number(hub.x) || 0) + Math.cos(localAngle) * localRadius;
          node.y = (Number(hub.y) || 0) + Math.sin(localAngle) * localRadius;
        });

        for (var iteration = 0; iteration < 120; iteration += 1) {
          var deltas = {};
          nodes.forEach(function(node) {
            deltas[node.id] = { x: 0, y: 0 };
          });
          for (var i = 0; i < nodes.length; i += 1) {
            for (var j = i + 1; j < nodes.length; j += 1) {
              var left = nodes[i];
              var right = nodes[j];
              var dx = (Number(right.x) || 0) - (Number(left.x) || 0);
              var dy = (Number(right.y) || 0) - (Number(left.y) || 0);
              var distanceSq = Math.max(0.012, dx * dx + dy * dy);
              var distance = Math.sqrt(distanceSq);
              var force = 0.018 / distanceSq;
              var fx = (dx / distance) * force;
              var fy = (dy / distance) * force;
              deltas[left.id].x -= fx;
              deltas[left.id].y -= fy;
              deltas[right.id].x += fx;
              deltas[right.id].y += fy;
            }
          }
          edges.forEach(function(edge) {
            var source = byId[edge.source];
            var target = byId[edge.target];
            var dx = (Number(target.x) || 0) - (Number(source.x) || 0);
            var dy = (Number(target.y) || 0) - (Number(source.y) || 0);
            var distance = Math.max(0.001, Math.sqrt(dx * dx + dy * dy));
            var ideal = hubIds[source.id] || hubIds[target.id] ? 0.32 : 0.46;
            ideal += (1 - Math.min(1, Number(edge.weight) || 0.4)) * 0.22;
            var pull = (distance - ideal) * 0.012;
            var fx = (dx / distance) * pull;
            var fy = (dy / distance) * pull;
            deltas[source.id].x += fx;
            deltas[source.id].y += fy;
            deltas[target.id].x -= fx;
            deltas[target.id].y -= fy;
          });
          var cooling = 0.095 * (1 - iteration / 160);
          nodes.forEach(function(node) {
            var radius = Math.sqrt((Number(node.x) || 0) * (Number(node.x) || 0) + (Number(node.y) || 0) * (Number(node.y) || 0));
            var targetRadius = hubIds[node.id] ? 0.34 : ((degree[node.id] || 0) < 0.72 ? 0.86 : 0.58);
            var central = node.type === "aesthetic" ? 0.018 : 0.010;
            var delta = deltas[node.id];
            if (radius > 0.001) {
              var radialForce = (radius - targetRadius) * central;
              delta.x -= ((Number(node.x) || 0) / radius) * radialForce;
              delta.y -= ((Number(node.y) || 0) / radius) * radialForce;
            }
            var nextX = (Number(node.x) || 0) + Math.max(-0.07, Math.min(0.07, delta.x * cooling));
            var nextY = (Number(node.y) || 0) + Math.max(-0.07, Math.min(0.07, delta.y * cooling));
            var nextRadius = Math.sqrt(nextX * nextX + nextY * nextY);
            var maxRadius = hubIds[node.id] ? 0.78 : 1.02;
            if (nextRadius > maxRadius) {
              nextX = (nextX / nextRadius) * maxRadius;
              nextY = (nextY / nextRadius) * maxRadius;
            }
            node.x = nextX;
            node.y = nextY;
          });
        }
        nodes.forEach(function(node) {
          var finalRadius = Math.sqrt((Number(node.x) || 0) * (Number(node.x) || 0) + (Number(node.y) || 0) * (Number(node.y) || 0));
          if (finalRadius > 1.02) {
            node.x = (Number(node.x) || 0) / finalRadius * 1.02;
            node.y = (Number(node.y) || 0) / finalRadius * 1.02;
          }
        });
        graph.layoutPrepared = true;
      }

      function resizeGraphCanvas() {
        var rect = graphFrame.getBoundingClientRect();
        if (!rect.width || !rect.height) return;
        var dpr = window.devicePixelRatio || 1;
        graphCanvas.width = Math.max(1, Math.floor(rect.width * dpr));
        graphCanvas.height = Math.max(1, Math.floor(rect.height * dpr));
        graphCanvas.style.width = rect.width + "px";
        graphCanvas.style.height = rect.height + "px";
        graphCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        drawGraph();
      }

      function resizeHarmonyCanvas() {
        var rect = harmonyCanvas.getBoundingClientRect();
        if (!rect.width || !rect.height) return;
        var dpr = window.devicePixelRatio || 1;
        harmonyCanvas.width = Math.max(1, Math.floor(rect.width * dpr));
        harmonyCanvas.height = Math.max(1, Math.floor(rect.height * dpr));
        harmonyCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        drawHarmony();
      }

      function projectGraph(node, width, height) {
        var scale = Math.min(width, height) * 0.50 * graphState.zoom;
        var motion = graphMotion(node);
        var base = graphAtomBase(node);
        return {
          x: width / 2 + graphState.panX + base.x * scale + motion.x,
          y: height / 2 + graphState.panY + base.y * scale + motion.y
        };
      }

      function inverseProjectGraph(x, y, width, height) {
        var scale = Math.min(width, height) * 0.50 * graphState.zoom;
        return {
          x: (x - width / 2 - graphState.panX) / scale,
          y: (y - height / 2 - graphState.panY) / scale
        };
      }

      function graphNodeRadius(node) {
        var base = 3.2 + Math.min(5.8, Math.max(0, Number(node.weight) || 0.4) * 5.8);
        if (node.type === "aesthetic") return Math.min(10.5, base + 1.3);
        if (node.type === "image") return Math.max(4.2, base - 0.6);
        return base;
      }

      function drawGraphGrid(width, height) {
        graphCtx.save();
        graphCtx.fillStyle = cssVar("--canvas-bg", "#191919");
        graphCtx.fillRect(0, 0, width, height);
        graphCtx.strokeStyle = cssVar("--canvas-grid", "rgba(248,248,248,0.08)");
        graphCtx.lineWidth = 1;
        var gap = 48;
        for (var x = 0; x <= width; x += gap) {
          graphCtx.beginPath();
          graphCtx.moveTo(x, 0);
          graphCtx.lineTo(x, height);
          graphCtx.stroke();
        }
        for (var y = 0; y <= height; y += gap) {
          graphCtx.beginPath();
          graphCtx.moveTo(0, y);
          graphCtx.lineTo(width, y);
          graphCtx.stroke();
        }
        graphCtx.restore();
      }

      function drawGraphAxes(width, height) {
        var centerX = width / 2 + graphState.panX;
        var centerY = height / 2 + graphState.panY;
        var axisLength = Math.min(width, height) * 0.34 * graphState.zoom;
        var axes = [
          { label: "x", color: cssVar("--canvas-electron", "#5D71FC"), x: axisLength, y: 0 },
          { label: "y", color: cssVar("--canvas-chart", "#f89540"), x: 0, y: axisLength * 0.66 },
          { label: "z", color: cssVar("--canvas-nucleus", "#EB5757"), x: -axisLength * 0.62, y: -axisLength * 0.46 }
        ];
        graphCtx.save();
        graphCtx.font = "12px Inter, system-ui, sans-serif";
        graphCtx.lineWidth = 1.4;
        axes.forEach(function(axis) {
          graphCtx.strokeStyle = colorToRgba(axis.color, 0.65);
          graphCtx.beginPath();
          graphCtx.moveTo(centerX, centerY);
          graphCtx.lineTo(centerX + axis.x, centerY + axis.y);
          graphCtx.stroke();
          graphCtx.fillStyle = axis.color;
          graphCtx.fillText(axis.label, centerX + axis.x + 5, centerY + axis.y + 4);
        });
        graphCtx.restore();
      }

      function drawGraphShells(width, height) {
        var centerX = width / 2 + graphState.panX;
        var centerY = height / 2 + graphState.panY;
        var radius = Math.min(width, height) * 0.39 * graphState.zoom;
        var yaw = graphState.motionTime * 0.00016;
        graphCtx.save();
        graphCtx.strokeStyle = cssVar("--canvas-shell", "rgba(93,113,252,0.18)");
        graphCtx.lineWidth = 1.2;
        graphCtx.setLineDash([6, 7]);
        [
          [radius, radius * 0.52, yaw],
          [radius * 0.82, radius * 0.36, -yaw * 1.15],
          [radius * 0.54, radius * 0.54 * 0.72, yaw + Math.PI * 0.52],
          [radius * 1.10, radius * 0.26, -yaw + Math.PI * 0.18]
        ].forEach(function(shell) {
          graphCtx.beginPath();
          graphCtx.ellipse(centerX, centerY, shell[0], shell[1], shell[2], 0, Math.PI * 2);
          graphCtx.stroke();
        });
        graphCtx.restore();
      }

      function drawGraphNucleus(width, height) {
        var centerX = width / 2 + graphState.panX;
        var centerY = height / 2 + graphState.panY;
        var color = cssVar("--canvas-nucleus", "#EB5757");
        graphCtx.save();
        var pulse = 1 + Math.sin(graphState.motionTime * 0.003) * 0.08;
        var gradient = graphCtx.createRadialGradient(centerX, centerY, 1, centerX, centerY, 24 * pulse);
        gradient.addColorStop(0, colorToRgba(color, 0.95));
        gradient.addColorStop(1, colorToRgba(color, 0));
        graphCtx.fillStyle = gradient;
        graphCtx.beginPath();
        graphCtx.arc(centerX, centerY, 24 * pulse, 0, Math.PI * 2);
        graphCtx.fill();
        graphCtx.fillStyle = color;
        graphCtx.beginPath();
        graphCtx.arc(centerX, centerY, 4.5 * pulse, 0, Math.PI * 2);
        graphCtx.fill();
        graphCtx.restore();
      }

      function drawGraph() {
        var rect = graphCanvas.getBoundingClientRect();
        if (!rect.width || !rect.height) return;
        graphCtx.clearRect(0, 0, rect.width, rect.height);
        drawGraphGrid(rect.width, rect.height);
        if (graphState.atomLayout) {
          drawGraphShells(rect.width, rect.height);
          drawGraphAxes(rect.width, rect.height);
          drawGraphNucleus(rect.width, rect.height);
        }
        if (!analysisPayload || !analysisPayload.graph || !analysisPayload.graph.nodes.length) {
          graphEmpty.hidden = false;
          return;
        }
        graphEmpty.hidden = true;
        var graph = analysisPayload.graph;
        var byId = {};
        graph.nodes.forEach(function(node) { byId[node.id] = node; });
        var visible = graphVisibleNodeSet(graph);
        var essentialEdges = graphEssentialEdgeMap(graph, visible);
        var focus = graphFocusSet(graph, visible, essentialEdges);
        graphCtx.save();
        graph.edges.forEach(function(edge) {
          if (!graphEdgeVisible(edge, visible, essentialEdges)) return;
          var source = byId[edge.source];
          var target = byId[edge.target];
          if (!source || !target) return;
          var a = projectGraph(source, rect.width, rect.height);
          var b = projectGraph(target, rect.width, rect.height);
          var connected = !focus || edge.source === focus.id || edge.target === focus.id;
          var forcedVisible = essentialEdges[graphEdgeKey(edge)] && Number(edge.weight) < graphState.minEdge;
          var edgeAlpha = focus ? (connected ? 0.40 + Number(edge.weight) * 0.28 : 0.025) : 0.10 + Number(edge.weight) * 0.16;
          if (forcedVisible && !focus) edgeAlpha = Math.max(edgeAlpha, 0.16);
          graphCtx.strokeStyle = colorToRgba("#f8f8f8", edgeAlpha);
          graphCtx.lineWidth = forcedVisible ? 0.55 : (focus && connected ? 0.9 + Number(edge.weight) * 0.85 : 0.45 + Number(edge.weight) * 0.75);
          graphCtx.beginPath();
          graphCtx.moveTo(a.x, a.y);
          if (graphState.atomLayout) {
            var seed = graphSeed(edge.source + edge.target + edge.type);
            var centerX = rect.width / 2 + graphState.panX;
            var centerY = rect.height / 2 + graphState.panY;
            var bend = ((seed % 100) / 100 - 0.5) * 0.32;
            var controlX = (a.x + b.x) / 2 + (centerX - (a.x + b.x) / 2) * (0.18 + bend);
            var controlY = (a.y + b.y) / 2 + (centerY - (a.y + b.y) / 2) * (0.18 - bend);
            graphCtx.quadraticCurveTo(controlX, controlY, b.x, b.y);
          } else {
            graphCtx.lineTo(b.x, b.y);
          }
          graphCtx.stroke();
        });
        graphCtx.restore();
        graph.nodes.slice().sort(function(left, right) {
          var leftBase = graphAtomBase(left);
          var rightBase = graphAtomBase(right);
          return leftBase.y - rightBase.y;
        }).forEach(function(node) {
          if (!visible[node.id]) return;
          var pos = projectGraph(node, rect.width, rect.height);
          var radius = graphNodeRadius(node);
          var color = graphColor(node);
          var selected = graphState.selectedNodeId === node.id || graphState.hoverNodeId === node.id;
          var focused = !focus || !!focus.nodes[node.id];
          graphCtx.save();
          graphCtx.globalAlpha = focused ? 1 : 0.16;
          if (selected) {
            var haloRadius = radius + 13;
            var halo = graphCtx.createRadialGradient(pos.x, pos.y, 1, pos.x, pos.y, haloRadius);
            halo.addColorStop(0, colorToRgba(color, 0.30));
            halo.addColorStop(1, colorToRgba(color, 0));
            graphCtx.fillStyle = halo;
            graphCtx.beginPath();
            graphCtx.arc(pos.x, pos.y, haloRadius, 0, Math.PI * 2);
            graphCtx.fill();
          }
          graphCtx.fillStyle = colorToRgba(color, focused ? 0.94 : 0.55);
          graphCtx.strokeStyle = selected ? colorToRgba("#f8f8f8", 0.96) : colorToRgba("#f8f8f8", focused ? 0.16 : 0.06);
          graphCtx.lineWidth = selected ? 1.8 : 0.75;
          graphCtx.beginPath();
          graphCtx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
          graphCtx.fill();
          graphCtx.stroke();
          if (node.type === "image") {
            graphCtx.strokeStyle = colorToRgba("#f8f8f8", selected ? 0.52 : 0.18);
            graphCtx.lineWidth = 0.7;
            graphCtx.beginPath();
            graphCtx.arc(pos.x, pos.y, radius + 3.2, 0, Math.PI * 2);
            graphCtx.stroke();
          }
          graphCtx.restore();
        });
        if (graphState.showLabels) drawGraphLabels(graph.nodes, visible, rect.width, rect.height, focus);
      }

      function animateGraph(timestamp) {
        graphAnimationFrame = null;
        graphState.motionTime = timestamp || performance.now();
        if (activeView === "graph") {
          drawGraph();
          drawHarmony();
          graphAnimationFrame = requestAnimationFrame(animateGraph);
        }
      }

      function startGraphAnimation() {
        if (!graphAnimationFrame) {
          graphAnimationFrame = requestAnimationFrame(animateGraph);
        }
      }

      function stopGraphAnimation() {
        if (graphAnimationFrame) {
          cancelAnimationFrame(graphAnimationFrame);
          graphAnimationFrame = null;
        }
      }

      function drawGraphLabels(nodes, visible, width, height, focus) {
        graphCtx.save();
        graphCtx.font = "11px Inter, system-ui, sans-serif";
        graphCtx.textBaseline = "middle";
        var placed = [];
        var labelLimit = nodes.length > 120 ? 18 : (nodes.length > 70 ? 24 : 34);
        var drawn = 0;
        nodes.slice().sort(function(left, right) {
          var leftActive = (graphState.selectedNodeId === left.id || graphState.hoverNodeId === left.id) ? 2 : 0;
          var rightActive = (graphState.selectedNodeId === right.id || graphState.hoverNodeId === right.id) ? 2 : 0;
          var leftType = left.type === "aesthetic" ? 0.8 : 0;
          var rightType = right.type === "aesthetic" ? 0.8 : 0;
          return (rightActive + rightType + Number(right.weight || 0)) - (leftActive + leftType + Number(left.weight || 0));
        }).forEach(function(node) {
          if (!visible[node.id]) return;
          if (node.type === "image") return;
          var active = graphState.selectedNodeId === node.id || graphState.hoverNodeId === node.id;
          var threshold = nodes.length > 80 ? 0.70 : 0.58;
          if (!active && Number(node.weight) < threshold) return;
          if (!active && drawn >= labelLimit) return;
          var pos = projectGraph(node, width, height);
          var label = String(node.label || "");
          var textWidth = graphCtx.measureText(label).width;
          var radius = graphNodeRadius(node);
          var box = {
            x0: pos.x + radius + 5,
            y0: pos.y - 8,
            x1: pos.x + radius + 9 + textWidth,
            y1: pos.y + 9
          };
          var overlaps = placed.some(function(other) {
            return box.x0 < other.x1 && box.x1 > other.x0 && box.y0 < other.y1 && box.y1 > other.y0;
          });
          if (overlaps && !active) return;
          placed.push(box);
          drawn += 1;
          graphCtx.fillStyle = colorToRgba("#f8f8f8", !focus || focus.nodes[node.id] ? 0.72 : 0.18);
          graphCtx.fillText(label, pos.x + radius + 6, pos.y);
        });
        graphCtx.restore();
      }

      function pickGraphNode(x, y) {
        if (!analysisPayload || !analysisPayload.graph) return null;
        var rect = graphCanvas.getBoundingClientRect();
        var visible = graphVisibleNodeSet(analysisPayload.graph);
        var best = null;
        var bestDistance = 18;
        analysisPayload.graph.nodes.forEach(function(node) {
          if (!visible[node.id]) return;
          var pos = projectGraph(node, rect.width, rect.height);
          var distance = Math.hypot(x - pos.x, y - pos.y);
          if (distance < bestDistance + graphNodeRadius(node)) {
            bestDistance = distance;
            best = node;
          }
        });
        return best;
      }

      function updateGraphTooltip(event, node) {
        if (!node) {
          graphTooltip.style.display = "none";
          graphTooltip.classList.remove("is-image-preview");
          return;
        }
        graphTooltip.classList.toggle("is-image-preview", node.type === "image");
        if (node.type === "image") {
          var index = imageIndexFromNode(node);
          var previewUrl = objectUrls[index];
          var size = graphImagePreviewSize();
          graphTooltip.style.setProperty("--preview-size", size + "px");
          graphTooltip.innerHTML = previewUrl ? '<img src="' + previewUrl + '" alt="">' : "";
        } else {
          graphTooltip.innerHTML = "<strong>" + escapeHtml(node.label) + "</strong>" + escapeHtml(node.type + " - cluster " + node.cluster + " - " + Math.round((node.weight || 0) * 100) + "%");
        }
        graphTooltip.style.display = "block";
        var frameRect = graphFrame.getBoundingClientRect();
        graphTooltip.style.transform = "translate(" + (event.clientX - frameRect.left + 14) + "px, " + (event.clientY - frameRect.top + 14) + "px)";
      }

      function imageIndexFromNode(node) {
        var match = /image-(\d+)/.exec(String(node.id || ""));
        return match ? Number(match[1]) - 1 : -1;
      }

      function graphImagePreviewSize() {
        var thumb = thumbGrid.querySelector(".thumb") || thumbGridNav.querySelector(".thumb");
        if (!thumb) return 72;
        var rect = thumb.getBoundingClientRect();
        return Math.max(48, Math.round(rect.width || 72));
      }

      function renderGraphDetail(node) {
        var detail = document.getElementById("graphDetail");
        if (!node) {
          detail.innerHTML = '<span>Selected node</span><strong>No node selected</strong><p>Click a node in the graph to inspect its cluster, weight and connected relations.</p>';
          return;
        }
        var graph = analysisPayload && analysisPayload.graph ? analysisPayload.graph : { edges: [] };
        var relations = (graph.edges || []).filter(function(edge) {
          return edge.source === node.id || edge.target === node.id;
        }).sort(function(a, b) {
          return Number(b.weight) - Number(a.weight);
        }).slice(0, 4);
        var extra = "";
        if (node.type === "image" && analysisPayload && analysisPayload.images) {
          var image = analysisPayload.images.find(function(item) { return item.id === node.id; });
          if (image) {
            extra = "Tags: " + (image.tags || []).slice(0, 5).join(", ");
          }
        }
        if (!extra && relations.length) {
          extra = "Relations: " + relations.map(function(edge) {
            return edge.type.replace(/_/g, " ") + " " + Math.round(Number(edge.weight) * 100) + "%";
          }).join(", ");
        }
        if (!extra) extra = "No visible relations at the current filter level.";
        detail.innerHTML = '<span>' + escapeHtml(node.type + " - cluster " + node.cluster) + '</span><strong>' + escapeHtml(node.label) + '</strong><p>Weight ' + Math.round(Number(node.weight || 0) * 100) + '%. ' + escapeHtml(extra) + '</p>';
      }

      function downloadBlob(blob, filename) {
        var url = URL.createObjectURL(blob);
        var link = document.createElement("a");
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        setTimeout(function() { URL.revokeObjectURL(url); }, 1500);
      }

      function exportGraphPng() {
        if (!analysisPayload) return;
        graphCanvas.toBlob(function(blob) {
          if (blob) downloadBlob(blob, "moodboard_graph.png");
        });
      }

      function exportGraphJson() {
        if (!analysisPayload) return;
        downloadBlob(new Blob([JSON.stringify(analysisPayload, null, 2)], { type: "application/json" }), "moodboard_analysis.json");
      }
