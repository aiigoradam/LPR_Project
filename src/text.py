
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate, plot_slice, plot_contour
import plotly

# Load the existing study
study_name = "Unet_Optimization"  # Change this to your study name
storage_url = "sqlite:///C:/Users/stopc/Desktop/LPR_Project/optuna_study.db"  # Path to the database storing your Optuna study
study = optuna.load_study(study_name=study_name, storage=storage_url)

# Display various visualizations
# 1. Optimization history plot - shows validation loss over time
opt_hist_fig = plot_optimization_history(study)
opt_hist_fig.show()

# 2. Hyperparameter importance plot - shows which parameters had the biggest impact on the objective
param_importance_fig = plot_param_importances(study)
param_importance_fig.show()

# 3. Parallel coordinate plot - shows relationships between parameters and objective values
parallel_coord_fig = plot_parallel_coordinate(study)
parallel_coord_fig.show()

# 4. Slice plot - shows how each parameter affects the objective, with best values highlighted
slice_fig = plot_slice(study)
slice_fig.show()

# 5. Contour plot - shows parameter interactions and their influence on the objective
contour_fig = plot_contour(study)
contour_fig.show()
