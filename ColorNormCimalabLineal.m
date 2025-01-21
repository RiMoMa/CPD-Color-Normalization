function [Inorm,H,E] = ColorNormCimalabLineal(Im,int,bins,Io, beta, alpha, HERef, maxCRef)

h = size(Im,1);
w = size(Im,2);

% transmitted light intensity
if ~exist('Io', 'var') || isempty(Io)
    Io = 220;
end

% OD threshold for transparent pixels
if ~exist('beta', 'var') || isempty(beta)
    beta = 0.15;
end

% tolerance for the pseudo-min and pseudo-max
if ~exist('alpha', 'var') || isempty(alpha)
    alpha = 1;
end

% reference H&E OD matrix
if ~exist('HERef', 'var') || isempty(HERef)
    HERef = [
    [0.8987    1.3352    0.5606];
    [0.1675    0.4757    0.1957]
        
    
        ];
   HERef = [
    [0.5   1    0.9];
    [0.1   0.6    0.1]
        
    
        ];
end

% reference maximum stain concentrations for H&E
if ~exist('maxCRef)', 'var') || isempty(maxCRef)
    maxCRef = [
    1.2784
    1.7246
    
        ];
end


%[Img]=uigetfile('*.bmp;*.jpg','*.tiff','IMAGENES');  
%Im=imread(Img);
%Im = imread('/media/ricardo/My Passport/internship/ATYPIA_classes_norm/clase_1/A12_00Ab.png');
Imagen = Im;

[R,G,B] = imsplit(Imagen);

R= R(:);
G= G(:);
B= B(:);
RGB=[R G B];
R_B= double(R)- double(B);
s=size(Imagen);
RGB=[R G B];
RGBCopy= RGB;
OD = -log(double((RGB+1))/Io);

% remove transparent pixels
ODhat = OD(~any(OD < beta, 2), :);

RGB = Io*exp(-ODhat);


%% Histogramas por Intervalos

[G_ordenado,indice]=sort(G);
R_B_ordenado=(R_B(indice));
R_ordenado= (R(indice));
B_ordenado= (B(indice));
%minG= min(G_ordenado);
%maxG= max(G_ordenado);

intervalos=int;

%limites= linspace(double(minG),double(maxG),intervalos);
limites=0:255/intervalos:255;

%hi= {};
histograma=[];
indice_menor=0;
rojo={};
azul={};
verde={};

for a=1:intervalos
    indice_mayor=limites(a+1);
    nombre=['Int ' num2str(indice_menor) ' a ' num2str(indice_mayor)];
    %indices_valor(i)= (indice_menor+indice_mayor)/2;
    
    indices=find(G_ordenado<indice_mayor & G_ordenado>indice_menor);
    %vector son los valores de R-B en cada intervalo de G
    vector=R_B_ordenado(indices);
    
    v_rojo= R_ordenado (indices);
    if isempty (v_rojo)
        v_rojo(a)=0;
    end
    
    v_azul=B_ordenado(indices);
    if isempty (v_azul)
        v_azul(a)=0;
    end
    
    v_verde= G_ordenado(indices);
    if isempty (v_verde)
        v_verde(a)=0;
    end
    
    rojo= [rojo; v_rojo'];
    azul=[azul; v_azul'];
    verde=[verde;v_verde'];
    
    if  isempty(vector)
        Mean_his(a)= 0;
        Mediana_his(a)=0;
        Std_his(a)=0;
        histo_ceros=zeros(1,bins);
        histograma=[histograma,histo_ceros];
        
    else
         
       %% Histograma
histo_img = zeros(1,bins); % variable con zeros donde bins es el numero de bins
rangoDatos = [-255,255];
TotalDatos = rangoDatos(2)-rangoDatos(1);
Variacion = TotalDatos/bins;

for occ = 1:bins % for para ir llenando los datos bin a bin
   freq = sum (vector>(rangoDatos(1)+Variacion*(occ-1)) & vector<=rangoDatos(1)+Variacion*(occ)  ); % determinar el numero de ocurrencias %aqui se debe definir un rango ; despues se suma
if any (vector<=(rangoDatos(1)+Variacion*(occ-1))) && occ == 1
    freq = freq+sum(vector<(rangoDatos(1)+Variacion*(occ-1)));
end
if any (vector>rangoDatos(1)+Variacion*(occ)) && occ == bins
    freq = freq+sum(vector>rangoDatos(1)+Variacion*(occ));
end
   histo_img(1,occ) = freq; % se pone el numero de ocurrencias en el bin correspondiente
end
          histograma=[histograma, histo_img];
        
        Mean_his (a)= mean(vector); %Valor Media cada Histograma
        Mediana_his(a)=median(vector); % Valor Mediana cada Histograma
        Std_his(a)= std(vector); % Desviacion estandar de la media
        
      
    end
    indice_menor=indice_mayor;
    
end

% Grafica de las medias
%         F7= figure(7);
%         F7.WindowState= 'maximized';
%         plot (Mean_his,'go-','LineWidth',2 )
%         xlabel('Intervals', 'FontSize',13,'FontWeight','bold')
%         ylabel('Mean','FontSize',13,'FontWeight','bold')
%         title(['Mean-Intervals  ' NomImage ])


%% Analisis de las curvas de las medias
%
%         %Derivadas
%         first_deri= diff(Mean_his);
%         second_deri=diff(first_deri);
%
%         % Primer Intervalo
%         [ffval, posff]=  sort(first_deri(1:5), 'descend');
%         [ minvalRB , posminRB]=sort(second_deri(1:5));

%Segundo Intervalo
%         seg_int=second_deri(8:end);
%         [maxvalRB, posmaxRB]=sort(seg_int);
%         posmaxRB= posmaxRB+8;  % se suman 8 para que las posiciones queden como las iniciales en la segunda derivada (intervalos (1:8) & (8:end )).

% Con la media solamente
[mincurvasRB, posmincurvasRB]=min(Mean_his(1:round(length(Mean_his)/2)));
%[mincurvasRB, posmincurvasRB]= min(Mean_his(1:(int/2)))

if mincurvasRB==0
    posval= find(Mean_his~=0);
    val= Mean_his(posval);
    minval= min(val);
    posmincurvasRB= find(Mean_his==minval);
end

maxcurvasRB= max(Mean_his(round(length(Mean_his)/2):int));
posmaxcurvasRB=find(Mean_his==maxcurvasRB);
posmaxcurvasRB= max(posmaxcurvasRB);
tamRB=size(Mean_his,1);

%Intervalos con la  segunda derivada

% j=posminRB(1); % Se supone que es la base de Hematoxilina (2)
%i=posmaxRB(1);  %Se supone que es la base de Eosina (23)

%Intervalos con la posicion  de las medias solamente
j=posmincurvasRB;
i=posmaxcurvasRB;

% Intervalos con la segunda derivada y la media
%         j=posminRB(1);
%         i=posmaxcurvasRB;


%% Se encuentras las bases de HEMATOXILINA usando el Intervalo [sminR, sminG sminB]
% Si no encuentra el intervalo vaya al siguiente
% Base Rojo
sminR= rojo{j,1};
x=j;
while sum(sminR)==0
    
    x=x+1;
    sminR= rojo{x,1};
    
end

%Base Verde

sminG= verde{j,1};
x=j;
while sum(sminG)==0
    x=x+1;
    sminG=verde{x};
end

%Base Azul
sminB= azul{j,1};
x=j;
while sum(sminB)==0
    x=x+1;
    sminB=azul{x};
end

% Se encuentras las bases de EOSINA usando el Intervalo [smaxR, smaxG smaxB]
%Si no encuentra el intervalo vaya al de atras
%Base para Rojo
smaxR= rojo{i,1};
y=i;
while  sum(smaxR)==0
    y=y-1;
    smaxR= rojo{y};
end

%Base para Verde
smaxG=verde{i,1};
y=i;
while sum(smaxG)==0
    y=y-1;
    smaxG=verde{y};
end

% Base para azul
smaxB=azul{i,1};
y=i;
while sum(smaxB)==0
    y=y-1;
    smaxB=azul{y};
end


%% Base de HEMATOXILINA
sminRGB=[sminR' sminG' sminB'];

if size(sminRGB,1)==1
    medminimosRGB=sminRGB;
else
    medminimosRGB=mean(sminRGB);
end

%% Base de EOSINA
smaxRGB=[smaxR' smaxG' smaxB'];

if size(smaxRGB,1)==1
    medmaximosRGB=smaxRGB;
else
    medmaximosRGB=mean(smaxRGB);
end

%% Matriz de las bases de HEMATOXILINA Y EOSINA.
A1=[medminimosRGB; medmaximosRGB]; % 

%A es la matriz que contiene las bases de hematoxilina y eosina
%A es usada para hacer la transformaci�n lineal usando el espacio de
%densidad �ptica OD

%% Espacio de Ddensidad optico
A=-log(double(A1)/255); 
if A(1,1) > A(2,1)
    A = [A(1,:); A(2,:)];
else
    A = [A(2,:) ; A(1,:)];
end

for ii=1:2
    for jj=1:3
        if isinf(A(ii,jj))
            A(ii,jj)=0;
        end
    end
end

%OD=-log(double(RGBCopy)/255);  % pixeles de rgb en OD

%HEod=OD*(pinv(A));  % este es C 

%Hod=HEod(:,1)*A(1,:);  
%Eod=HEod(:,2)*A(2,:);

%H=255*exp(-Hod);
%E=255*exp(-Eod);

%EE es la imagen de eosina y HH es la de hematoxilina

%He=reshape(H,s(1),s(2),3);
%HH=uint8(He);

%Eo=reshape(E,s(1),s(2),3);
%EE=uint8(Eo);

%nivel=graythresh(HH);
%imgBW1 = im2bw(HH,nivel);
%imgBW1=~imgBW1;

%se = strel('disk',1);
%imgBWe=imerode(imgBW1,se);
%imgBWeo=imopen(imgBWe,se);
%imgBWeo=imerode(imgBWeo,se);
%imgBWeof=imfill(imgBWeo,'holes');
%BW1=imopen(imgBWeof,se);

%BW1 es la imagen binaria (segementada con Otsu y desp�es de
%algunas operaciones morfologicas)

%L1=bwlabel(BW1);

%         figure(8)
%         subplot(2,2,1)
%         imshow(Imagen);title('Original image')
%         subplot(2,2,2)
%         imshow(HH);title('Hematoxylin Image')
%         subplot(2,2,3)
%         imshow(EE);title('Eosin Image')
%         subplot(2,2,4)
%         imshow(L1);title('Segmented Image')


    
OD=-log(double(RGBCopy)/255);

HEod=OD*(pinv(A));  

% normalize stain concentrations
maxC = prctile(HEod', 99, 2);

HEod = bsxfun(@rdivide, HEod', maxC);
HEod = bsxfun(@times, HEod', maxCRef');
%perturbar H

% recreate the image using reference mixing matrix
Inorm = Io*exp(-HERef' * HEod');
Inorm = reshape(Inorm', h, w, 3);
Inorm = uint8(Inorm);

Hod=HEod(:,1)*HERef(1,:);
Eod=HEod(:,2)*HERef(2,:);

H=Io*exp(-Hod);
E=Io*exp(-Eod);

%EE es la imagen de eosina y HH es la de hematoxilina


He=reshape(H,h,w,3);
HH=uint8(He);

Eo=reshape(E,h,w,3);
EE=uint8(Eo);
H = HH;
E = EE;
end