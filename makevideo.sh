#!/usr/bin/env bash

ffmpeg -framerate 10 -i result/KITTI/01/ORB2/frameimage/0022/%06d.jpg -vcodec libx264 -pix_fmt yuv420p -r 10 result/KITTI/01/ORB2/frameimage/0022_video.mp4
