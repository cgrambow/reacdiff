function turing_pp_video(index,video_file,data_file,fps)
%run after turing_pp
%index is the frame to be saved in video
%video_file is the filepath of the video
%data_file is the filepath of the data
if nargin<3
   data_file = '~/Dropbox (MIT)/2.168 Project/Data/turing_ss';
end
if nargin<4
  fps = 30;
end
mat = load(data_file);
y = mat.y;

yout = permute(squeeze(y(index,:,:,end,1)),[2,3,4,1]);
yout = yout/max(yout(:)); 
myMovie = VideoWriter(video_file);
set(myMovie,'FrameRate',fps);
open(myMovie);
writeVideo(myMovie,yout);
close(myMovie);
