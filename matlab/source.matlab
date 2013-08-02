-------------------------------------------------
%matlab code
function TestTileF2()
%% -----Initialiation-----
L=30500;%pinhole camera distance
drx=0;
dry=0;

lambda=0.780;%laser wavelength
z0=100;
bin1=1;
pixelsize=5.2*bin1;%size of camera pixel
pixelnum=512/bin1;%number of pixels for the reconstruction
bkground=0;

img1=imread('1cmlens1sttest.bmp','bmp');%imports the background image file
(i.e the screen image when no object is present)
if bkground==1
img2=imread('cc.bmp','bmp');%imports the object hologram (i.e the screen
image when the object is present)
else
img2=0;
end

img1=img1-img2;
img1=img1(1:(pixelnum*bin1),1:(pixelnum*bin1));
if bin1~=1
for ii=1:pixelnum
    for jj=1:pixelnum
      Scatter1(ii,jj)=mean(mean(img1(((ii-1)*bin1+1):((ii)*bin1),((jj-1)*bin1+1):((jj)*bin1))));
    end
end
else
  Scatter1=double(img1);
end

img1=img1-mean(mean(img1));

%%
%  -----------------geometry-------------------------------
k=(2*pi)/lambda;%wavevector of laser
camerawidth=pixelsize*pixelnum/2;
NumericalA=camerawidth/(L^2+camerawidth^2)^.5
koff=k*NumericalA;
T=2^ceil(log2(koff*pixelsize/pi))*4
%newpixelsize=pixelsize*(pixelnum/T-1)/(pixelnum-1);
x=(1:pixelnum)*pixelsize;
x=x-mean(x);
y=x;

[xx,yy]=meshgrid(x,y);


%%
%-------------------------Scatterer-----------------------------------
%% -------------------------do tile algorithm

Esum1=0;
pixelnum1=pixelnum/T;
%%
newpixelsize=pixelsize/T;
dx=(pixelsize-newpixelsize)/2; %the small bit that make the tile algorith
accurate
tic %start clock
for ii=1:T
    for jj=1:T
Esub1=Scatter1(((ii-1)*pixelnum1+1):(ii*pixelnum1),((jj-1)*pixelnum1+1):(jj*pixelnum1));

for mm=1:pixelnum1
    for nn=1:pixelnum1
tile1(((mm-1)*T+1):((mm)*T),((nn-1)*T+1):((nn)*T))=Esub1(mm,nn); %here the
simplest interpolate is used, and it works!
% post filtering at line (***) is important
    end
end

xstart=xx(((ii-1)*pixelnum1+1),((jj-1)*pixelnum1+1))-dx;
xend=xx(ii*pixelnum1,jj*pixelnum1)+dx;
ystart=yy(((ii-1)*pixelnum1+1),((jj-1)*pixelnum1+1))-dx;yend=yy(ii*pixelnum1,jj*pixelnum1)+dx;


XI=linspace(xstart,xend,pixelnum); YI=linspace(ystart,yend,pixelnum);
[XXI,YYI]=meshgrid(XI,YI);
R1I=((XXI+drx).^2+(YYI+dry).^2+L^2).^.5; %making the r vectors
E1refI=exp(1i*k*R1I)./R1I;
Eprop1=tile1.*E1refI;
Esum1=Esum1+Eprop1;
    end
end
toc
Ek1=fft2(Esum1);%take fast fourier transform of Esub so it is represented
in kspace
kpixelsize=((2*pi)/newpixelsize/pixelnum);
kx=((0:(pixelnum-1))-pixelnum/2)*kpixelsize; ky=kx;
kx=fftshift(kx);ky=fftshift(ky); [kkx,kky]=meshgrid(kx,ky);%makes a matrix
from kx and ky

Ek1=Ek1.*(1+sign(koff^2-kkx.^2))/2.*(1+sign(koff^2-kky.^2))/2; % post
filtering: remove the noise outside the Numerical aperture ---- (***)

scanrange=2*z0;%this is the range from z0 that the reconstructions will be
made at. the reconstruction are made either side of z0 so this is actually
half the full range
h=scanrange/2;%gives the step size

n=1;%defines integer n that can be increased in a loop
ttt1=zeros(1,n);%defines an empty matrix to store the max(mean(intensity))
of all the reconstructed images
zzz=zeros(1,n);%defines an empty matrix to store z.
for zob1=(-scanrange/2):h:(+scanrange/2)%loops zob from -scanrange to
+scanrange in h steps
    u=L-zob1;%object distance from lens (or camera screen)
    phi1=real((1+sign(k^2-kkx.^2-kky.^2)).*((k^2-kkx.^2-kky.^2).^.5)*u/2);
    phi1=mod(phi1,2*pi);
    Ek11=Ek1.*exp(-1i*phi1);% propagates Ek1 a distance v to obtain Ek
    Ex1=ifft2(Ek11);%takes a fast fourier transform of Ek to change it to
x,y,z space
    Ex1=ifftshift(ifftshift(Ex1,1),2);
    tt1=abs(Ex1.^2);%calculates the intensity of Ex
    ttt1(:,n)=max(mean(tt1));%stores the max mean intensity of each Ex and
stores it in a matrix

    zzz(:,n)=(L-u); %distance from pinhole in um.
    figure(2)
%   
imagesc(XI-mean(XI),YI-mean(YI),tt1(1:(pixelnum),(1):(pixelnum)));%image a
scaled version of the intensity of Ex
    imagesc(XI-mean(XI),YI-mean(YI),tt1);%image a scaled version of the
intensity of Ex
%    figure(2)
%    imagesc(XI-mean(XI),YI-mean(YI),tt2);%image a scaled version of the
intensity of Ex
    xlabel('x/um');%labels x and y axes
    ylabel('y/um');
    ll=sprintf('image-pinhole distance=%3.1f um', zob1);%distance
    title(ll);%puts title on the graph
    drawnow
    n=n+1;%increases n by one so the next figure will have a different
number to this one and allows ttt to be plotted against n.
end
figure(3)
plot(zzz,ttt1);
%% plot the reconstructed image at the best focus
[m,n]=max(ttt1);
u=L-zzz(n);
    phi1=real((1+sign(k^2-kkx.^2-kky.^2)).*((k^2-kkx.^2-kky.^2).^.5)*u/2);
%The (1+sign())/2 factor ensures the phase factor is real
    phi1=mod(phi1,2*pi);
    Ek11=Ek1.*exp(-1i*phi1);% propagates Ek1 a distance v to obtain Ek
    Ex1=ifft2(Ek11);%takes a fast fourier transform of Ek to change it to
x,y,z space
    Ex1=ifftshift(ifftshift(Ex1,1),2);
    tt1=abs(Ex1.^2);%calculates the intensity of Ex
    ttt1(:,n)=max(max(tt1));%stores the max mean intensity of each Ex and
stores it in a matrix

    zzz(:,n)=(L-u); %distance from pinhole in um.
    figure(2)
%   
imagesc(XI-mean(XI),YI-mean(YI),tt1(1:(pixelnum),(1):(pixelnum)));%image a
scaled version of the intensity of Ex
%    imagesc(XI-mean(XI),YI-mean(YI),tt1);%image a scaled version of the
intensity of Ex
    h=mesh(XXI-mean(XI),YYI-mean(YI),tt1);%mesh
    view([0 0 1])
    axis tight
     set(h,'FaceColor','interp','EdgeColor','none')
    xlabel('x/um');%labels x and y axes
    ylabel('y/um');
    ll=sprintf('image-pinhole distance=%3.1f um', L-u);%distance
    title(ll);%puts title on the graph

