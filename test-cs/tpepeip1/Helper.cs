using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;

namespace tpepeip1
{
    public static class Helper
    {
        public const int BPM_LOW = 50;
        public const int BPM_HIGH = 180;
        public const int BPM_NOISE_HIGH = 120;

        public static readonly Scalar RED = new Scalar(0, 0, 255);
        public static readonly Scalar GREEN = new Scalar(0, 255, 0);
        public static readonly Scalar BLUE = new Scalar(255, 0, 0);
        public static readonly Scalar WHITE = new Scalar(255, 255, 255);
        public static readonly Scalar BLACK = new Scalar(0, 0, 0);
    }

    public delegate void TextDlg(string text, int x, int y);
}
