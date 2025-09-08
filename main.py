import argparse
import wandb

from src.solver.solver import Solver

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento de PINN con PyTorch Lightning y Wandb")
    parser.add_argument('--train', action='store_false', help='Entrena el modelo')
    parser.add_argument('--input_dim', type=int, default=3, help='Dimensión de entrada')
    parser.add_argument('--n_hidden', type=int, default=5, help='Número de capas ocultas')
    parser.add_argument('--dim_hidden', type=int, default=64, help='Dimensión de las capas ocultas')
    parser.add_argument('--output_dim', type=int, default=2, help='Dimensión de salida')
    parser.add_argument('--epochs', type=int, default=315000, help='Número máximo de épocas')
    parser.add_argument('--batch_size', type=int, default=512, help='Tamaño de lote')
    parser.add_argument('--lr', type=float, default=5e-4, help='Tasa de aprendizaje')
    parser.add_argument('--ratio', type=float, default=0.1, help='Ratio de datos')
    parser.add_argument('--seed', type=int, default=1, help='Número de pasos de MP')
    parser.add_argument('--wandb_project', type=str, default='AP- Marta', help='Nombre del proyecto en wandb')
    parser.add_argument('--wandb_name', type=str, default='AP- Marta -comparison', help='Nombre del experimento en wandb')
    parser.add_argument('--factor_ph', type=float, default=0.014829614927386174, help='Physics loss coefficient')
    parser.add_argument('--factor_ic', type=float, default=0.00002899560424848756, help='Initial condition loss coefficient')
    parser.add_argument('--factor_bc', type=float, default=0.00334113787602771, help='Boundary condition loss coefficient')
    parser.add_argument('--validation_freq', type=int, default=1, help='Frecuencia de valiación')
    parser.add_argument('--scheduler_type', type=str, default='plateau', choices=['cosine', 'plateau'], help='Tipo de scheduler para la tasa de aprendizaje')
    parser.add_argument('--test_gif_freq', type=int, default=500, help='Frecuencia de generación de GIFs durante la prueba')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Número de épocas de calentamiento para el scheduler')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    solver = Solver(args)
    if args.train:
        solver.train()
    
    # Test the model
    solver.test()
    
if __name__ == '__main__':
    main()

