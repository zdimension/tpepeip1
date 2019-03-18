using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using static tpepeip1.Logger;

namespace tpepeip1
{
    public class Camera
    {
        public bool Connected { get; private set; }
        public Size CameraShape { get; private set; }
        public VideoCapture TheCamera { get; private set; }
        public string Path { get; private set; }

        private Camera(VideoCapture camera)
        {
            TheCamera = camera;

            try
            {
                Info("Trying to connect to camera " + Path);
                var tmp = new Mat();
                var status = TheCamera.Read(tmp);
                Info("Got after read");
                CameraShape = tmp.Size();
                Connected = true;
            }
            catch
            {
                CameraShape = Size.Zero;
                Connected = false;
                Error("Couldn't connect to camera " + Path);
            }
        }

        public Camera(int id) : this(new VideoCapture(id))
        {
            Path = id.ToString();
        }

        public Camera(string path) : this(new VideoCapture(path))
        {
            Path = path;
        }

        public void Dispose()
        {
            TheCamera.Release();
        }

        public Mat Frame()
        {
            if (!Connected)
                throw new ArgumentException("Trying to read frame from invalid camera " + Path);

            var res = new Mat();

            try
            {
                TheCamera.Read(res);
            }
            catch
            {
                Error("An error occured while reading the frame");
                throw;
            }

            return res;
        }
    }
}
