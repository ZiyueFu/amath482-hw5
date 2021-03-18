clear all, clc, close all
% % Find the number of frames that represents snapshots in each time series
% % (delta t)
% v = VideoReader('ski_drop_low.mp4');
% slices = v.NumberOfFrames;
% Snapshot = [];
% % The matrix snapshot has a dimension m*n where m is pixel of image and
% % n is number of images
% while hasFrame(v)
%     img = rgb2gray(readFrame(v));
%     img = reshape(img, [540*960], 1);
%     Snapshot = [Snapshot img];
% end
% Snapshot = double(Snapshot);
% t2 = linspace(0,v.CurrentTime,slices+1);
% t = t2(1:slices);
% dt = t(2) - t(1);
% %%
% % Create DMD Matrices
% X1 = Snapshot(:,1:end-1);
% X2 = Snapshot(:,2:end);
% % SVD of X1
% [U, Sigma, V] = svd(X1, 'econ');
% figure(1)
% plot(diag(Sigma)/sum(diag(Sigma)),'m*')
% xlabel('Frames')
% ylabel('Singular Value')
% title('Singular Value Spectrum of Ski')
% % We got the rank is one, then we rebuild the U, S, V
% rank = 1;
% U = U(:, 1:rank);
% Sigma = Sigma(1:rank, 1:rank);
% V = V(:, 1:rank);
% % Computation of ~S
% S = U'*X2*V*diag(1./diag(Sigma));
% [eV, D] = eig(S); % compute eigenvalues + eigenvectors
% mu = diag(D); % extract DMD eigenvalues
% omega = log(mu)/dt;
% Phi = U*eV;
% %%
% % Create DMD Solution
% y0 = Phi\X1(:,1); % pseudoinverse to get initial conditions
% modes = zeros(length(y0),length(t));
% for iter = 1:length(t)
%    modes(:,iter) = y0.*exp(omega*t(iter)); 
% end
% X_LR_dmd = Phi*modes;
% 
% % %%
% % % DMD Solution
% % waterfall(x,1:slices,abs(Phi')), colormap([0 0 0])
% % xlabel('x')
% % ylabel('modes')
% % zlabel('|u|')
% % title('DMD Modes')
% % set(gca,'FontSize',16)
% %%
% % X_SP_dmd = Snapshot - abs(X_LR_dmd);
% X_SP = Snapshot - X_LR_dmd;
% X_SP = X_SP + 0.000002;
% R = X_SP .* (X_SP<0);
% X_SP_dmd = X_SP - R;
% 
% %%
% figure(2)
% for img = 1:slices
%     subplot(1, 3, 1)
%     original = uint8(Snapshot(:,img));
%     imshow(reshape(original, 540, 960))
%     title('Original Video')
%     subplot(1, 3, 2)
%     back = uint8(X_LR_dmd(:,img));
%     imshow(reshape(back, 540, 960))
%     title('Background Video')
%     subplot(1, 3, 3)
%     fore = real(X_SP_dmd(:,img));
%     imshow(reshape(fore, 540, 960))
%     title('Foreground Video')
%     drawnow
% end
%%
v = VideoReader('monte_carlo_low.mp4');
slices = v.NumberOfFrames;
%%
Snapshot = [];
% The matrix snapshot has a dimension m*n where m is pixel of image and
% n is number of images
while hasFrame(v)
    img = rgb2gray(readFrame(v));
    img = reshape(img, [540*960], 1);
    Snapshot = [Snapshot img];
end
Snapshot = double(Snapshot);
t2 = linspace(0,v.CurrentTime,slices+1);
t = t2(1:slices);
dt = t(2) - t(1);
%%
% Create DMD Matrices
X1 = Snapshot(:,1:end-1);
X2 = Snapshot(:,2:end);
% SVD of X1
[U, Sigma, V] = svd(X1, 'econ');
figure(1)
plot(diag(Sigma)/sum(diag(Sigma)),'m*')
xlabel('Frames')
ylabel('Singular Value')
title('Singular Value Spectrum of Monte Carlo')
%%
% We got the rank is one, then we rebuild the U, S, V
rank = 2;
U = U(:, 1:rank);
Sigma = Sigma(1:rank, 1:rank);
V = V(:, 1:rank);
% Computation of ~S
S = U'*X2*V*diag(1./diag(Sigma));
[eV, D] = eig(S); % compute eigenvalues + eigenvectors
mu = diag(D); % extract DMD eigenvalues
omega = log(mu)/dt;
Phi = U*eV;
%%
% Create DMD Solution
y0 = Phi\X1(:,1); % pseudoinverse to get initial conditions
modes = zeros(length(y0),length(t));
for iter = 1:length(t)
   modes(:,iter) = y0.*exp(omega*t(iter)); 
end
X_LR_dmd = Phi*modes;

%%
% % DMD Solution
% waterfall(x,1:slices,abs(Phi')), colormap([0 0 0])
% xlabel('x')
% ylabel('modes')
% zlabel('|u|')
% title('DMD Modes')
% set(gca,'FontSize',16)
%%
% X_SP_dmd = Snapshot - abs(X_LR_dmd);
X_SP = Snapshot - X_LR_dmd;
% X_SP = X_SP + 0.000002;
R = X_SP .* (X_SP<0);
X_SP_dmd = X_SP - R;

%%
figure(2)
for img = 1:slices
    subplot(1, 3, 1)
    original = uint8(Snapshot(:,img));
    imshow(reshape(original, 540, 960))
    title('Original Video')
    subplot(1, 3, 2)
    back = uint8(X_LR_dmd(:,img));
    imshow(reshape(back, 540, 960))
    title('Background Video')
    subplot(1, 3, 3)
    fore = real(X_SP_dmd(:,img));
    imshow(reshape(fore, 540, 960))
    title('Foreground Video')
    drawnow
end