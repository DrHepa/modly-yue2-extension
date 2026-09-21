# Tutorial práctico: YuE2 en Modly

Este tutorial explica el uso básico de los nodos YuE2 y, sobre todo, cómo preparar un **cover**. Está basado en la guía oficial de YuE2:

- [YuE2 Music Skill](https://github.com/multimodal-art-projection/YuE/tree/main/skills/yue2-music)
- [Generation and covers](https://github.com/multimodal-art-projection/YuE/blob/main/skills/yue2-music/references/generation-and-covers.md)
- [Models, setup and audio-to-score](https://github.com/multimodal-art-projection/YuE/blob/main/skills/yue2-music/references/models-and-setup.md)
- [ABC editing](https://github.com/multimodal-art-projection/YuE/blob/main/skills/yue2-music/references/abc-editing.md)

## Antes de empezar

1. Instala la extensión desde **Modly → Extensions → Install from GitHub**.
2. Ejecuta **Repair/setup** y descarga los modelos desde la interfaz de modelos de Modly.
3. Comprueba primero **Installation Diagnostics**.
4. Usa `device=cuda` en una GPU compatible. La generación completa puede tardar y consume bastante VRAM.

Los resultados se guardan en `Workflows/YuE2/` dentro del workspace. Los nodos que producen texto suelen devolver un **bundle descriptor** JSON: no es el ABC directamente. Para extraer el ABC hay que usar **Inspect Bundle → abc**.

## Qué hace cada nodo

| Nodo | Para qué sirve |
|---|---|
| **Generate Song** | Genera una canción desde letra y estilo. `cot=full` crea melodía y armonía; `cot=melody` crea una melodía sin acordes; `cot=off` genera directamente sin plan editable. |
| **Plan Score** | Genera y guarda solamente el plan ABC y sus artefactos. Es la mejor entrada para editar la composición antes de renderizarla. |
| **Render ABC / Cover / Edit** | Recibe ABC en su entrada de texto y lo convierte en una canción nueva con la letra y el estilo indicados en sus parámetros. Usar `cot=melody` para un cover melódico y `cot=full` para reharmonizar. |
| **Render Exact Saved Plan** | Reutiliza un plan guardado sin volver a tokenizar su ABC. No usarlo después de editar el ABC. |
| **Inspect Bundle / Agent Package** | Extrae `abc`, `request`, `metadata`, `descriptor`, `verify` o un paquete de edición. |
| **Audio to Score / Artifacts** | Inspecciona un audio/bundle YuE2 ya producido. No transcribe por sí solo una canción externa. |
| **Import Native Saved Artifacts** | Importa una carpeta nativa con `plan.json`, `latent.npy` u otros artefactos y la convierte en descriptor reutilizable. |
| **Plan to Semantic Tokens** | Continúa desde un plan y ejecuta la generación semántica. |
| **Semantic Tokens to Latents** | Continúa desde tokens semánticos y produce latentes acústicos. |
| **Decode Audio Latents** | Convierte latentes guardados en WAV/FLAC usando la VAE. |
| **Encode Audio** | Codifica un WAV/FLAC de 48 kHz estéreo en latentes VAE. No hace transcripción ni convierte audio en ABC. |
| **Batch Requests / Resume** | Ejecuta varias peticiones JSON/JSONL secuencialmente y reutiliza elementos completados. |
| **Installation Diagnostics** | Comprueba Python, dependencias, modelos y almacenamiento sin generar música. |

## Workflow básico: generar una canción editable

Conecta los nodos así:

```text
Text / Lyrics
      │
      ▼
Generate Song (text → audio)
      │
      └── audio para escuchar
```

Configuración recomendada:

- `input_mode=lyrics`
- `cot=full`
- `style`: género, instrumentos, idioma, voz y tempo aproximado
- `lyrics`: letra con secciones, por ejemplo `[Verse]`, `[Chorus]`, `[Bridge]`
- `seed`: fijo si querés comparar versiones

Para conservar el plan editable, usa este flujo:

```text
Lyrics → Plan Score → Inspect Bundle (abc) → Render ABC / Cover / Edit → Audio
```

No edites `plan.json`, `prefix.npy` ni `abc_tokens.npy`. Copiá el `score.abc`, editá la copia y enviála a **Render ABC** como una petición nueva.

## Workflow básico de cover desde una melodía ABC

Este es el flujo de cover que se puede hacer directamente con esta extensión:

```text
Melody-only ABC ───────────────┐
                               ▼
                         Render ABC / Cover / Edit ──► Audio cover
                               ▲
                         lyrics + style
```

### Paso a paso en Modly

1. Añadí un nodo **Text/File input** y cargá un archivo `.abc` de melodía.
2. Añadí **Render ABC / Cover / Edit**.
3. Conectá la salida de texto del primer nodo a la entrada de texto del nodo YuE2.
4. En el nodo YuE2 configurá:
   - `cot=melody`
   - `lyrics` con la letra que querés cantar
   - `lyrics_file` si preferís cargarla desde `.txt`
   - `style` con el nuevo género, instrumentación, idioma, tipo de voz y tempo
   - `seed` fijo para comparar estilos
5. Conectá la salida `audio` a un reproductor o a un nodo de guardado.
6. Ejecutá el workflow y revisá duración, pronunciación, entradas de voz y final de la canción.

### Qué significa `cot=melody`

`cot=melody` le pide a YuE2 que respete la melodía simbólica suministrada, pero deja libertad para crear el acompañamiento y la interpretación vocal. No conserva la voz original, la grabación ni el timbre del cantante.

El ABC debe ser principalmente melódico y no debe contener acordes si se busca un cover con acompañamiento nuevo. El tempo, compás, tonalidad y duración de las notas sí forman parte de la condición musical.

## Cover desde un audio existente: el paso que ocurre fuera de YuE2

YuE2 **no acepta un WAV como referencia de cover directa**. El VAE de **Encode Audio** tampoco transcribe melodía: solamente codifica audio en latentes acústicos.

La ruta oficial es:

```text
Audio original
      │
      ▼
SheetSage2 / transcriptor externo
      │
      ▼
Revisar y corregir score.abc
      │
      ▼
Eliminar acordes → melody-only.abc
      │
      ▼
Modly: Render ABC / Cover / Edit (cot=melody)
      │
      ▼
Cover con nuevo estilo y letra
```

En la práctica:

1. Transcribí el audio original con SheetSage2 usando la tarea de melodía completa.
2. Revisá la transcripción: errores de tempo, notas, silencios y compás son frecuentes.
3. Exportá una versión sin símbolos de acordes.
4. Cargá ese `.abc` en el input de texto de Modly.
5. En **Render ABC / Cover / Edit**, poné la nueva letra y el estilo objetivo.
6. Generá el cover y compará la melodía, las transiciones y la pronunciación.

La extensión no instala SheetSage2 automáticamente porque tiene dependencias y modelos distintos. Esto es intencional: la transcripción y la regeneración son dos etapas separadas y verificables.

## Rehacer un cover cambiando solamente el estilo

Conservá el mismo ABC, letra y seed, y cambiá solo `style`:

```text
melody-only.abc
      ├── Render ABC: cinematic pop
      ├── Render ABC: acoustic folk
      └── Render ABC: synthwave
```

Así la comparación tiene una temática común. No esperes ondas idénticas: YuE2 vuelve a generar la interpretación completa, aunque la condición melódica sea la misma.

## Editar armonía o estructura

Para cambiar acordes, usá un ABC con melodía y armonía y seleccioná `cot=full`:

```text
Plan Score
   ▼
Inspect Bundle (abc)
   ▼
Editar una copia de score.abc
   ▼
Render ABC / Cover / Edit (cot=full)
   ▼
Nueva versión completa
```

Antes de generar, comprobá que no se hayan alterado accidentalmente las notas, duraciones, compás o tempo que querías conservar. La edición de ABC genera una canción nueva; no es inpainting ni conserva muestras del audio original.

## Problemas habituales

- **El nodo recibe audio, pero espera texto**: para un cover usá el ABC transcrito, no el WAV.
- **La letra no se aplica**: en Render ABC la entrada conectada es el ABC; la letra va en `lyrics` o `lyrics_file`.
- **Se obtiene un JSON en vez del ABC**: conectá el descriptor a **Inspect Bundle** y seleccioná `extract=abc`.
- **Se perdió la melodía del cover**: revisá el ABC y usá `cot=melody`; `cot=full` puede rehacer la planificación armónica.
- **Se modificó un plan exacto**: no edites sus archivos internos; copiá `score.abc` y usá Render ABC.
- **La canción queda corta**: aumentá los límites de tokens respetando la VRAM y comprobá `result.json` para detectar truncamiento.

## Checklist de una prueba real

- [ ] Audio original conservado.
- [ ] ABC transcrito revisado.
- [ ] Versión melody-only guardada.
- [ ] Letra y estilo documentados.
- [ ] Seed documentado.
- [ ] Cover generado con `cot=melody`.
- [ ] Duración y truncamiento comprobados.
- [ ] Escucha humana de melodía, letra, transiciones y final.
- [ ] ABC, request, metadata y audio guardados juntos.

