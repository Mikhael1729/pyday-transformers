# Transformers

En este repositorio se encuentran los recursos para seguir la charla acerca de Transformers en PyDay. Los commits en el mismo se encuentran organizados en el mismo orden de la charla. Si deseas observar la construcción de la arquitectura que se realizará durante la charla, solo debes de revertir el historial de commits de manera cronológica, desde el más antiguo hasta el más reciente para ver su implementación desde cero.

## Dependencias

Debes tener instalado [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

## Instalación

1. Descarga e instala las dependencias usando conda:

    ```bash
    conda env create -f environment.yml
    ```

2. Activa las dependencias ejecutando:

    ```bash
    conda activate transformers-pyday
    ```

> Si usas VSCode, es muy probable que desees que el editor reconozca las dependencias que acabas de instalar. Para hacerlo, solo corre el comando `Ctrl + Shift + P` (o `Cmd + Shift + P` en macOS) y selecciona la opción "Select Interpreter"; dentro, escoge `transformers-pyday`. De esta forma, el IntelliSense de VSCode puede reconocer dependencias como PyTorch en el código.

## Ejecución del código

Para ejecutar el código, utiliza:

```bash
python main.py
```