# YuE2 para Modly — resumen de entrega

Se entrega el código fuente de una extensión `process`, versión **0.1.0 candidata**, con **13 nodos** y una suite local de **52 pruebas superadas**. Repositorio de la extensión: [DrHepa/modly-yue2-extension](https://github.com/DrHepa/modly-yue2-extension). No se ha hecho inferencia con los pesos reales.

## Qué contiene

Generación de canciones desde letra y estilo; planificación `full`, `melody` y `off`; entrada ABC para covers o partituras editadas; recuperación exacta de un plan; generación semántica; síntesis de latentes; codificación y decodificación VAE; extracción de partitura y metadatos; paquete de edición para un agente externo; importación de artefactos nativos; lotes con reutilización verificada de elementos completos; diagnósticos.

Las etapas internas se conectan mediante descriptores JSON transportados por puertos `text`. Los resultados musicales salen como `audio`, con archivos persistentes en el workspace. Esto evita inventar tipos de puerto que Modly 0.4.2 no tiene.

## Instalación y pesos

`setup.py` acepta el JSON de Modly y su forma posicional antigua. Usa el ejecutable exacto que proporciona el host, tanto si es CPython 3.11 público como 3.12 privado. Crea `<extensión>/venv`; no cambia el Python de Modly ni el del sistema.

Descarga el código de YuE2 fijado a un commit y lo verifica. Instala dependencias aisladas, comprueba importaciones/API/kernels pequeños y después descarga **YuE2-3B, YuE2-Vae y YuE2-Vae-legacy** en el `models_dir` real de Modly. Los trece nodos y ambas lanes comparten esos snapshots. Los pesos no están dentro del ZIP, la extensión o cada nodo por separado. Reparar o actualizar el código no implica volver a descargar pesos íntegros.

La selección de plataformas contempla Windows x86_64, Linux x86_64 y Linux ARM64. Las rutas están implementadas y se han revisado los wheels publicados, pero aún no están probadas de extremo a extremo. La ruta CUDA fija PyTorch 2.10.0+cu128. El backend vLLM es opcional y solo se ofrece para Linux CUDA.

## Límites que no hay que confundir

Un **cover desde ABC** está implementado. La transcripción automática de un audio a ABC es un componente externo, por ejemplo SheetSage2, que no se ha integrado en este paquete. Su entorno tiene requisitos diferentes y no conviene mezclarlo con el de YuE2.

La **edición asistida por agente** tiene un paquete de entrada/salida, pero no se incorpora un agente autónomo. La persona o el agente modifica partitura, letra o estilo, y YuE2 regenera la canción. No es una edición localizada del waveform ni separación de voz e instrumentos.

La codificación VAE produce **latentes acústicos**: no extrae partitura, letra ni tokens semánticos de covers. Exige entrada estéreo a 48 kHz; no cambia discretamente el audio por detrás.

## Cambios en Modly documentados

El cambio importante para prometer una cancelación correcta es implementar la terminación de procesos Python activos: en el runner inspeccionado, `terminate()` está vacío. También conviene pasar `modelsDir` explícitamente al setup y al runtime, y reenviar stderr de forma acotada y en vivo. Un visor de múltiples artefactos y un editor ABC son mejoras opcionales.

No se ha modificado Modly ni abierto un issue o PR. Las ubicaciones del código, los cambios mínimos y las pruebas de aceptación están detallados en `MODLY_GAPS.md`.

## Qué significa “52 pruebas superadas”

Se ejecutaron en un contenedor **Linux x86_64, Python 3.13.5 y PyTorch 2.10.0 CPU**. Comprueban el contrato, las rutas, los parámetros, el protocolo, la integridad y el encadenado con dobles de prueba. **No certifican las lanes 3.11/3.12, Windows, ARM64 ni CUDA.** El flujo CI y los scripts para hacer las pruebas reales están incluidos, pero no se han ejecutado en esas máquinas.

La siguiente prueba de aceptación concreta es instalar el paquete en una máquina objetivo, ejecutar `check_native_api.py` y después `smoke.py` con el Python del venv de la extensión. La tabla completa de validación pendiente está en `VALIDATION.md`.

Las licencias también están separadas: wrapper MIT; código YuE2 Apache-2.0; checkpoints CC BY-NC 4.0. La licencia del wrapper no elimina las condiciones de los pesos.
