import cudf
from cuopt import routing
cost_matrix = cudf.DataFrame([[0,2,2,2],[2,0,2,2],[2,2,0,2],[2,2,2,0]], dtype='float32')
task_locations = cudf.Series([1,2,3])
n_vehicles = 2
dm = routing.DataModel(cost_matrix.shape[0], n_vehicles, len(task_locations))
dm.add_cost_matrix(cost_matrix)
dm.add_transit_time_matrix(cost_matrix.copy(deep=True))
ss = routing.SolverSettings()
sol = routing.Solve(dm, ss)
print(sol.get_route())
print('\n\n****************** Display Routes *************************')
sol.display_routes()