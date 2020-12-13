from imports import *

class FlickrSet(td.Dataset):
    def __init__(self, root_dir='flickr30k', image_size=(256, 256), mode='train', sigma=40):
        super(FlickrSet, self).__init__()
        self.sigma = sigma
        
        self.root_dir = root_dir
        self.mode = mode
        self.images_dir = os.path.abspath(os.path.join(root_dir, mode))
        self.files = os.listdir(self.images_dir)
        self.image_size = image_size
        
        if mode == 'train':
            self.files = self.files[:1000]
        elif mode == 'test':
            self.files = self.files[:600]
    
    def __len__(self):
        return len(self.files)
    
    def __repr__(self):
        return "FlickrSet(mode={}, sigma={})".format(self.mode, self.sigma)

    def __getitem__(self, idx):        
        img_path = os.path.join(self.images_dir, self.files[idx])
        clean = Image.open(img_path).convert('L')
        
        i = np.random.randint(clean.size[0] - self.image_size[0]) 
        j = np.random.randint(clean.size[1] - self.image_size[1])
                
        transform = tv.transforms.Compose([tv.transforms.RandomCrop(self.image_size),
                                           tv.transforms.ToTensor()])
        
        clean = transform(clean)
        
        '''
        add AWGN directly to image
        '''
        noise = torch.FloatTensor(clean.size()).normal_(mean=0, std=self.sigma/255.)        
        noisy = clean + noise
        
        return noisy, clean, noise.squeeze()
        

class DnCNN(nn.Module):
    def __init__(self, num_of_layers, channels=1):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(spectral_norm(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(spectral_norm(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(spectral_norm(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False)))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out


class Denoiser():
    def __init__(self, net, experiment_name, sigma, batch_size, data=False):        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.batch_size = batch_size
        self.sigma = sigma
        
        self.train_set = None
        self.test_set = None
        self.train_loader = None
        self.test_loader = None
        
        if data:
            self.train_set = FlickrSet(mode='train', sigma=self.sigma, image_size=(40,40))
            self.test_set = FlickrSet(mode='test', sigma=self.sigma, image_size=(40,40))
        
            self.train_loader = td.DataLoader(self.train_set, batch_size=batch_size, shuffle=True,
                                              drop_last=True, pin_memory=True)
            self.test_loader = td.DataLoader(self.test_set, batch_size=batch_size, shuffle=False,
                                             drop_last=True, pin_memory=True)

        self.history = []
        self.train_loss = []
        self.train_psnr = []
        
        self.net = net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        eval('self.criterion.' + self.device + '()')

        output_dir = 'checkpoints/' + experiment_name
        os.makedirs(output_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(output_dir, "checkpoint.pth.tar")
        self.config_path = os.path.join(output_dir, "config.txt")

        # Transfer all local arguments/variables into attributes
        locs = {k: v for k, v in locals().items() if k != 'self'}
        self.__dict__.update(locs)

        # Load checkpoint and check compatibility
        if os.path.isfile(self.config_path):
            with open(self.config_path, 'r') as f:
                if f.read()[:-1] != repr(self):
                    raise ValueError(
                        "Cannot create this experiment: "
                        "I found a checkpoint conflicting with the current setting.")
            self.load()
        else:
            self.save()

    @property
    def epoch(self):
        return len(self.history)

    def setting(self):
        return {'Net': self.net,
                'TrainSigma': self.sigma,
                'Optimizer': self.optimizer,
                'BatchSize': self.batch_size}
    
    def __repr__(self):
        string = ''
        for key, val in self.setting().items():
            string += '{}({})\n'.format(key, val)
        return string

    def state_dict(self):
        return {'Net': self.net.state_dict(),
                'Optimizer': self.optimizer.state_dict(),
                'History': self.history,
                'TrainLoss' : self.train_loss, 
                'TrainPSNR' : self.train_psnr}

    def load_state_dict(self, checkpoint):
        self.net.load_state_dict(checkpoint['Net'])
        self.optimizer.load_state_dict(checkpoint['Optimizer'])
        
        self.history = checkpoint['History']
        self.train_loss = checkpoint['TrainLoss']
        self.train_psnr = checkpoint['TrainPSNR']

    def save(self):
        torch.save(self.state_dict(), self.checkpoint_path)
        with open(self.config_path, 'w') as f:
            print(self, file=f)

    def load(self):
        checkpoint = torch.load(self.checkpoint_path,
                                map_location=self.device)
        self.load_state_dict(checkpoint)
        del checkpoint

    def myimshow(self, image, ax):
        image = image.cpu().detach().numpy()
        image = (image - image.min())/(image.max() - image.min())
        h = ax.imshow(image.squeeze(), cmap='gray')
        return h
        
    def plot(self, fig, axes, noisy_img):
        with torch.no_grad():
            noisy_img = noisy_img.to(self.device)
            
            est_noise = self.net(noisy_img[None])[0]
            denoised = noisy_img - est_noise
            
        axes[0][0].clear()
        axes[0][1].clear()
        axes[1][0].clear()
        axes[1][1].clear()
                
        self.myimshow(noisy_img, ax=axes[0][0])
        axes[0][0].set_title('Noisy Image')
        
        self.myimshow(denoised, ax=axes[0][1])
        axes[0][1].set_title('Denoised Image')
        
        axes[1][0].plot([self.train_loss[k] for k in range(self.epoch)], label='training loss')
        axes[1][0].legend(loc='upper right')
        axes[1][0].set(xlabel='Epoch', ylabel='Loss')
        
        axes[1][1].plot([self.train_psnr[k] for k in range(self.epoch)], label='training psnr')
        axes[1][1].legend(loc='lower right')
        axes[1][1].set(xlabel='Epoch', ylabel='PSNR')
        
        fig.tight_layout()
        fig.canvas.draw()
        
        
    def run(self, num_epochs, fig, axes, noisy_img):
        self.net.train()
        start_epoch = self.epoch
        print("Start/Continue training from epoch {}".format(start_epoch))
        
        self.plot(fig, axes, noisy_img)
                    
        num_updates = 0
        running_loss = 0
        running_psnr = 0
        
        for epoch in range(start_epoch, num_epochs):            
            for noisy, clean, noise  in self.train_loader:
                noisy, clean, noise = noisy.to(self.device), clean.to(self.device), noise.to(self.device)
                
                self.optimizer.zero_grad()
                est_noise = self.net(noisy).squeeze()
                                
                loss = self.criterion(est_noise, noise)
                
                loss.backward()
                self.optimizer.step()
                
                with torch.no_grad():
                    num_updates += 1
                    
                    running_loss += loss.item()
                    
                    est_clean = noisy - self.net(noisy)
                    est_clean = (est_clean - est_clean.min())/(est_clean.max() - est_clean.min())
                    
                    running_psnr += 10*torch.log10(len(est_clean.reshape(-1))/torch.norm(est_clean-clean)**2)
            
            self.history.append(epoch)
            self.train_loss.append(running_loss/num_updates)
            self.train_psnr.append(running_psnr/num_updates)
                
            print("Epoch {}, Train PSNR {} ".format(self.epoch, self.train_psnr[-1]))
            self.save()
            
            self.plot(fig, axes, noisy_img)
                        
        print("Finish training for {} epochs".format(num_epochs))

    def evaluate(self):
        self.net.eval()
        
        with torch.no_grad():
            num_updates = 0
            running_loss = 0
            running_psnr = 0
            
            for noisy, clean, noise in self.test_loader:                
                noisy, clean, noise = noisy.to(self.device), clean.to(self.device), noise.to(self.device)
                
                est_noise = self.net(noisy).squeeze()
                loss = self.criterion(est_noise, noise)
                
                num_updates += 1
                
                running_loss += loss.item()
                
                est_clean = noisy - self.net(noisy)
                est_clean = (est_clean - est_clean.min())/(est_clean.max() - est_clean.min())
                
                running_psnr += 10*torch.log10(len(est_clean.reshape(-1)) / torch.norm(est_clean-clean)**2)
                        
        self.net.train()
        
        return {'loss' : running_loss/num_updates, 'psnr' : running_psnr/num_updates}
    
    
