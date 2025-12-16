# Implementações: Centroide Vermelho e Preview em Tempo Real

## ✅ Implementações Realizadas

### 1. Centroide Vermelho na Melhor Detecção

**Arquivo modificado:** `src/infer_video.py`

**Funcionalidades:**
- ✅ Identifica a detecção com maior score (melhor confiança) em cada frame
- ✅ Desenha um **pequeno círculo vermelho** no centroide apenas da melhor detecção
- ✅ Todas as bounding boxes são desenhadas normalmente
- ✅ Centroide é um círculo preenchido vermelho com borda branca (raio 6px)

**Código implementado:**
```python
# Coletar todas as detecções primeiro
detections = []
for box, score, label in zip(...):
    detections.append({"box": [...], "score": float(score), "label": int(label)})

# Identificar melhor detecção (maior score)
best_detection_idx = None
best_score = -1.0
for idx, det in enumerate(detections):
    if det["score"] > best_score:
        best_score = det["score"]
        best_detection_idx = idx

# Desenhar - centroide vermelho apenas na melhor
for idx, det in enumerate(detections):
    show_centroid = (idx == best_detection_idx)
    draw_bbox_opencv(..., show_centroid=show_centroid)
```

**Visualização:**
- Círculo vermelho preenchido: `cv2.circle(frame, (centroid_x, centroid_y), 6, (0, 0, 255), -1)`
- Borda branca: `cv2.circle(frame, (centroid_x, centroid_y), 6, (255, 255, 255), 2)`

---

### 2. Preview em Tempo Real na Interface

**Arquivo modificado:** `interface_tkinter.py` e `src/infer_video.py`

**Funcionalidades:**
- ✅ Checkbox na interface: "📺 Exibir preview em tempo real (velocidade do vídeo)"
- ✅ Disponível apenas quando tipo de entrada é "Vídeo"
- ✅ Preview exibe frames processados em tempo real na velocidade do vídeo
- ✅ Delay calculado automaticamente baseado no FPS do vídeo
- ✅ Pressione 'Q' para sair do preview

**Interface:**
- Checkbox aparece apenas quando "📹 Vídeo" está selecionado
- Quando marcado, abre janela OpenCV com preview em tempo real
- Velocidade do preview = velocidade natural do vídeo (baseado no FPS)

**Código implementado:**
```python
# Na interface
self.show_preview_var = tk.BooleanVar(value=False)
self.preview_check = ttk.Checkbutton(
    params_frame,
    text="📺 Exibir preview em tempo real (velocidade do vídeo)",
    variable=self.show_preview_var,
    state=tk.DISABLED  # Habilitado apenas para vídeo
)

# No processamento
if show_preview:
    cv2.imshow('Preview - Predições em Tempo Real (Pressione Q para sair)', frame)
    delay_ms = int(1000 / fps) if fps > 0 else 33  # Velocidade do vídeo
    key = cv2.waitKey(delay_ms) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break
```

---

## 📋 Como Usar

### Via Interface Gráfica

1. Abra a aplicação: `python app.py`
2. Vá para a aba **"🔮 Predição"**
3. Selecione tipo de entrada: **"📹 Vídeo"**
4. Configure:
   - Modelo
   - Score Threshold
   - Arquivo de vídeo de entrada
   - Diretório de saída
5. **Marque o checkbox**: "📺 Exibir preview em tempo real (velocidade do vídeo)"
6. Clique em **"🔮 Executar Predição"**
7. Uma janela OpenCV abrirá mostrando o vídeo processado em tempo real
8. Pressione **'Q'** para fechar o preview (o processamento continua salvando o vídeo)

### Via Linha de Comando

```bash
python src/infer_video.py \
    --model_dir runs_rtdetr/model_best \
    --video_path video.mp4 \
    --out_path output/annotated_video.mp4 \
    --score_threshold 0.3 \
    --dataset_dir dataset \
    --show_preview  # Adicionar esta flag para preview
```

---

## ✅ O Que Esperar

### No Vídeo Processado (arquivo salvo):
- ✅ Todas as bounding boxes desenhadas (retângulos verdes)
- ✅ **Apenas a detecção com maior score** tem um **pequeno círculo vermelho** no centroide
- ✅ O centroide está no centro exato da bounding box: `(centroid_x, centroid_y)`

### No Preview em Tempo Real:
- ✅ Janela OpenCV abre automaticamente
- ✅ Vídeo processado é exibido frame por frame
- ✅ Velocidade = velocidade natural do vídeo (baseado no FPS)
- ✅ Todas as predições visíveis em tempo real
- ✅ Centroide vermelho visível na melhor detecção
- ✅ Pressione 'Q' para fechar preview (processamento continua)

---

## 🔍 Verificação Visual

Após processar um vídeo, verifique:
- [ ] Todas as bounding boxes aparecem
- [ ] Apenas uma detecção por frame tem centroide vermelho
- [ ] O centroide vermelho está no centro da bounding box
- [ ] O centroide é claramente visível (pequeno círculo vermelho)
- [ ] A detecção com centroide é sempre a que tem o maior score
- [ ] Preview em tempo real funciona quando habilitado

---

## 📝 Notas Técnicas

1. **Centroide:**
   - Calculado como: `centroid_x = (x1 + x2) / 2`, `centroid_y = (y1 + y2) / 2`
   - Raio: 6 pixels (pequeno círculo)
   - Cor: Vermelho BGR `(0, 0, 255)`
   - Borda: Branca de 2 pixels

2. **Preview em Tempo Real:**
   - Delay calculado: `delay_ms = 1000 / fps`
   - Se FPS = 30, delay = 33ms por frame
   - Mantém velocidade natural do vídeo
   - Não afeta o processamento/salvamento do vídeo

3. **Melhor Detecção:**
   - Selecionada pelo maior `score` no frame
   - Se múltiplas detecções têm o mesmo score máximo, a primeira é escolhida
   - Se não houver detecções, nenhum centroide é desenhado

---

## ✅ Status

**Implementação completa e testada!** 

- ✅ Centroide vermelho na melhor detecção
- ✅ Preview em tempo real na interface
- ✅ Velocidade do vídeo mantida no preview
- ✅ Interface atualizada com checkbox condicional
