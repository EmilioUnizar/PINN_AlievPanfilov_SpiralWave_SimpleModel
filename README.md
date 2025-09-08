# PINN Aliev-Panfilov Spiral Wave – Simple Model

Simulación de arritmia cardíaca utilizando **Redes Neuronales Físicamente Informadas (PINNs)** sobre el modelo de **Aliev-Panfilov**.  
El proyecto permite explorar la dinámica de ondas espirales en tejidos cardíacos mediante entrenamiento con **PyTorch Lightning**.

---

## 🚀 Instalación

Clona este repositorio y asegúrate de tener Python 3.9+:

```bash
git clone https://github.com/EmilioUnizar/PINN_AlievPanfilov_SpiralWave_SimpleModel.git
cd PINN_AlievPanfilov_SpiralWave_SimpleModel
pip install -r requirements.txt
```

Dependencias principales:
- [PyTorch Lightning](https://www.pytorchlightning.ai/)  
- [PyTorch](https://pytorch.org/)  
- NumPy  

---

## ▶️ Uso

Ejecutar una simulación estándar:

```bash
python main.py
```

Lanzar un barrido de parámetros definido en `config.yaml`:

```bash
python sweep.py
```

---

## ⚙️ Configuración

Los parámetros del barrido se encuentran en `config.yaml`:

- `factor_ph`  
- `factor_bc`  
- `factor_ic`  

---

## 📊 Resultados esperados

En esta sección se añadirán más adelante gráficas y visualizaciones de las simulaciones (ondas espirales, dinámicas de propagación, etc.).

---

## 🤝 Contribuir

Las contribuciones son bienvenidas.  
Si encuentras un problema o quieres proponer mejoras, abre un **issue** o envía un **pull request**.

---

## 📜 Licencia

Este proyecto se distribuye bajo la licencia MIT.  
