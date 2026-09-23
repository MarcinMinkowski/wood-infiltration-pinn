import torch
from torch import nn
import math
from model import NN

class Trainer:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = NN()
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-3)

        self.model.to(self.device)

        self.rho = 0.1
        self.sr = 0.2
        self.phi = 0.6
        self.Lambda_x = 0.96
        self.Lambda_y = 9.6e-5
        self.Lambda_z = 0.24

        self.lambda_initial = 1.0
        self.lambda_boundary = 1.0

        self.alpha = 0.9
        self.eta = 1e-3

        point_density = 100
        grid = torch.linspace(0,1,point_density,device=self.device)
        grid1, grid2, grid3 = torch.meshgrid([grid,grid,grid])

        grid1_flattened = torch.flatten(grid1)
        grid2_flattened = torch.flatten(grid2)
        grid3_flattened = torch.flatten(grid3)

        self.initial_points = torch.stack([grid1_flattened,grid2_flattened,grid3_flattened,torch.zeros(point_density**3,device=self.device)],axis=-1)

        grid1, grid2, grid_t = torch.meshgrid([grid,grid,grid[1:]])

        grid1_flattened = torch.flatten(grid1)
        grid2_flattened = torch.flatten(grid2)
        grid_t_flattened = torch.flatten(grid_t)

        bottom_points = torch.stack([grid1_flattened,grid2_flattened,torch.zeros(point_density**2*(point_density-1),device=self.device),grid_t_flattened],axis=-1)
        top_points = torch.stack([grid1_flattened,grid2_flattened,torch.ones(point_density**2*(point_density-1),device=self.device),grid_t_flattened],axis=-1)
        left_points = torch.stack([torch.zeros(point_density**2*(point_density-1),device=self.device),grid1_flattened,grid2_flattened,grid_t_flattened],axis=-1)
        right_points = torch.stack([torch.ones(point_density**2*(point_density-1),device=self.device),grid1_flattened,grid2_flattened,grid_t_flattened],axis=-1)
        front_points = torch.stack([grid1_flattened,torch.zeros(point_density**2*(point_density-1),device=self.device),grid2_flattened,grid_t_flattened],axis=-1)
        back_points = torch.stack([grid1_flattened,torch.ones(point_density**2*(point_density-1),device=self.device),grid2_flattened,grid_t_flattened],axis=-1)

        self.boundary_points = torch.cat([bottom_points,top_points,left_points,right_points,front_points,back_points])

    def generate_initial_points(self,n_points):
        initial_points = torch.rand((n_points,4),device=self.device)
        initial_points[:,3] = 0

        return initial_points
    
    def generate_boundary_points(self,n_points):
        boundary_points = torch.rand((n_points,4),device=self.device)
        idx = torch.arange(n_points,device=self.device)
        boundary_axis = torch.randint(0,3,(n_points,),device=self.device)
        boundary_side = torch.randint(0,2,(n_points,),device=self.device).float()
        boundary_points[idx,boundary_axis] = boundary_side

        return boundary_points

    def residual(self, X):
        u = self.model(X)
    
        u_grad = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
        u_x = u_grad[:,0:1]
        u_y = u_grad[:,1:2]
        u_z = u_grad[:,2:3]
        u_t = u_grad[:,3:4]
    
        u_x_grad = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_y_grad = torch.autograd.grad(u_y, X, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        u_z_grad = torch.autograd.grad(u_z, X, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]
    
        u_xx = u_x_grad[:,0:1]
        u_yy = u_y_grad[:,1:2]
        u_zz = u_z_grad[:,2:3]

        S = 0.5*(torch.erf((-self.rho/u-torch.tensor(1.0))/(math.sqrt(2)*self.sr))-torch.erf((self.rho-torch.tensor(1.0))/(math.sqrt(2)*self.sr)))
        C = self.rho/(self.sr*math.sqrt(2*math.pi)*u**2)*torch.exp(-(-self.rho/u-1.0)**2/(2*self.sr**2))
        a = torch.minimum(torch.tensor(1),8*10**4*X[:,3:4])
        kr = 1 + a*(S-1)

        return self.phi*C*u_t - kr*(self.Lambda_x*u_xx+self.Lambda_y*u_yy+self.Lambda_z*u_zz) - a*C*(self.Lambda_x*u_x**2+self.Lambda_y*u_y**2+self.Lambda_z*u_z**2)

    def train_update(self):
        points_PINN = torch.rand(100,4,device=self.device).requires_grad_()
        initial_points = self.generate_initial_points(100)
        boundary_points = self.generate_boundary_points(100)
    
        res = self.residual(points_PINN)
        initial = self.model(initial_points)
        boundary = self.model(boundary_points)
    
        loss_pinn = self.loss_fn(res,torch.zeros_like(res))
        loss_initial = self.loss_fn(initial,torch.full_like(initial,-1.0))
        loss_boundary = self.loss_fn(boundary,torch.full_like(boundary,-3.33e-11))

        grads_pinn = torch.autograd.grad(loss_pinn, self.model.parameters(), retain_graph=True)
        grads_initial = torch.autograd.grad(loss_initial, self.model.parameters(), retain_graph=True)
        grads_boundary = torch.autograd.grad(loss_boundary, self.model.parameters(), retain_graph=True)

        grads_pinn_flattened = []
        grads_initial_flattened = []
        grads_boundary_flattened = []

        for grad in grads_pinn:
            grads_pinn_flattened.append(torch.flatten(grad))

        grads_pinn_flattened = torch.cat(grads_pinn_flattened)
        grads_pinn_flattened_abs = torch.abs(grads_pinn_flattened)
        grad_pinn_max = torch.max(grads_pinn_flattened_abs).item()

        for grad in grads_initial:
            grads_initial_flattened.append(torch.flatten(grad))
            
        grads_initial_flattened = torch.cat(grads_initial_flattened)
        grads_initial_flattened_abs = torch.abs(grads_initial_flattened)
        grad_initial_mean = torch.mean(grads_initial_flattened_abs).item()

        for grad in grads_boundary:
            grads_boundary_flattened.append(torch.flatten(grad))
            
        grads_boundary_flattened = torch.cat(grads_boundary_flattened)
        grads_boundary_flattened_abs = torch.abs(grads_boundary_flattened)
        grad_boundary_mean = torch.mean(grads_boundary_flattened_abs).item()

        lambda_initial_new = grad_pinn_max/grad_initial_mean
        lambda_boundary_new = grad_pinn_max/grad_boundary_mean

        self.lambda_initial = (1-self.alpha)*self.lambda_initial+self.alpha*lambda_initial_new
        self.lambda_boundary = (1-self.alpha)*self.lambda_boundary+self.alpha*lambda_boundary_new

        loss = loss_pinn + self.lambda_initial*loss_initial + self.lambda_boundary*loss_boundary
    
        loss.backward()
    
        self.optimizer.step()
        self.optimizer.zero_grad()

        print(f"Initial conditions weight: {self.lambda_initial}, boundary conditions weight: {self.lambda_boundary}")
        print(f"Residual loss: {loss_pinn.item()}, initial loss: {loss_initial.item()}, boundary loss: {loss_boundary.item()}")

    def train_loop(self,n_epochs):
        for epoch in range(n_epochs):
            print(f"Epoch {epoch+1}:")
            self.train_update()

    def save_weights(self):
        torch.save(self.model.state_dict(),"weights.pt")
