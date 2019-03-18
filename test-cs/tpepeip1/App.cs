using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using static tpepeip1.Logger;
using static tpepeip1.Helper;

namespace tpepeip1
{
    public class App
    {
        public const float SIZE_NORMAL = 1.0f;
        public const float SIZE_SMALL = 0.75f;
        public const float SIZE_BIG = 1.5f;
        public const HersheyFonts FONT_SERIF = HersheyFonts.HersheyComplex;
        public const HersheyFonts FONT_SERIF_SMALL = HersheyFonts.HersheyComplexSmall;

        public List<Camera> USBCameras = new List<Camera>();

        public Camera Camera;

        public int Width = 0;
        public int Height = 0;
        public int Channels = 3;
        public bool ShowInfos = true;
        public ImageProcessor Processor;
        public float FontSize = 1;
        public Scalar TextColor;
        public float Zoom = 1;
        public Dictionary<char, Action> Keys;

        public App(Dictionary<string, string> cmdargs)
        {
            Info("Starting app with args " + cmdargs);

            {
                var cam = new Camera(0);
                USBCameras.Add(cam);

                Camera = cam;
            }

            Processor = new ImageProcessor("haarcascade_frontalface_alt.xml");
            TextColor = WHITE;
            
            Keys = new Dictionary<char, Action>
            {
                {'l', () => Processor.LockToggle() }
            };
        }

        public void Text(string text, int x, int y, Scalar? col=null, float size = SIZE_NORMAL,
            HersheyFonts font = FONT_SERIF, Mat frame = null)
        {
            if (frame == null)
                frame = Processor.Output;

            for(int ox = -1; ox <= 1; ox++)
            for(int oy = -1; oy <= 1; oy++)
                Cv2.PutText(frame, text, new Point(x + ox, y + oy), font, FontSize * size, BLACK, 2, LineTypes.AntiAlias);

            Cv2.PutText(frame, text, new Point(x, y), font, FontSize * size, col ?? TextColor, 2, LineTypes.AntiAlias);

        }

        public int TextRow = 0;
        public void Print(string text, Mat frame = null, Scalar? col = null)
        {
            Text(text, 20, 30 + TextRow * 30, col, 1, FONT_SERIF_SMALL, frame);
            TextRow++;
        }

        public void AppLoop()
        {
            var frame = Camera.Frame();

            Width = frame.Width;
            Height = frame.Height;
            Channels = frame.Channels();

            TextRow = 0;

            Processor.Input = frame;
            //try
            {
                Processor.Execute((a, b, c) => Text(a, b, c));
            }
            

            DisplayInfos();

            Cv2.ImShow("processed", Processor.Output);
        }

        public void DisplayInfos()
        {
            Text($"{Processor.CorrectBPM:F0} bpm", 15, 50, size:SIZE_BIG);
            Print($"{Processor.FPS:F0} fps");
        }
    }
}
