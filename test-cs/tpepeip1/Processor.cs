using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using MathNet.Numerics;
using OpenCvSharp;
using static tpepeip1.Helper;

namespace tpepeip1
{
    public class ImageProcessor
    {
        public const int SHIFT_THRESHOLD = 5;
        public const int MIN_SAMPLE_COUNT = 15;
        public const float FPS_SMOOTHING = 0.8f;

        public int LastFPS = 0;
        public int FPS = 0;
        public int BufferSize;
        public SlidingBuffer<double> AverageColorBuffer;
        public bool LockFace = false;
        public Mat Input = new Mat(new []{10, 10}, MatType.CV_8U);
        public Mat Output = new Mat(new []{10, 10}, MatType.CV_8U);
        public SlidingBuffer<Mat> InputBuffer;
        public bool Denoise = false;
        public bool Colorify = false;
        public CascadeClassifier Classifier;
        public SlidingBuffer<TimeSpan> TimesBuffer;
        public DateTime StartTime = DateTime.Now;
        public Rect Face = OpenCvSharp.Rect.Empty;
        public Point2f LastCentre = new Point2f(0, 0);
        public double[] Frequencies = { };
        public float[] Fourier = { };
        public double BPM = 0;
        public int Gap = 0;
        public int BufferState = 0;
        public List<double> Deviation = new List<double>();
        public SlidingBuffer<double> BPMBuffer;
        public double CorrectBPM = 0;
        public int BPMPosition = 0;
        public bool HighBPMFilter = true;
        public int CurrentHigh = BPM_NOISE_HIGH;


        public ImageProcessor(string cascade, int buffer_size = 256)
        {
            Classifier = new CascadeClassifier(cascade);
            AverageColorBuffer = new SlidingBuffer<double>(buffer_size);
            TimesBuffer = new SlidingBuffer<TimeSpan>(buffer_size);
            BPMBuffer = new SlidingBuffer<double>(buffer_size) { 60 };
            InputBuffer = new SlidingBuffer<Mat>(5);
            BufferSize = buffer_size;
        }

        public void ClearBuffers()
        {
            AverageColorBuffer.Clear();
            TimesBuffer.Clear();
            BPMBuffer.Clear();
        }

        public void Rect(Rect r, Scalar col)
        {
            Cv2.Rectangle(Output, r, col, 2);
        }

        public bool LockToggle()
        {
            return LockFace = !LockFace;
        }

        public bool DenoiseToggle()
        {
            return Denoise = !Denoise;
        }

        public bool HighBPMToggle()
        {
            CurrentHigh = HighBPMFilter ? BPM_HIGH : BPM_NOISE_HIGH;
            return HighBPMFilter = !HighBPMFilter;
        }

        public bool ColorifyToggle()
        {
            return Colorify = !Colorify;
        }

        public double CalcShift(Rect face)
        {
            var centre = new Point(face.X + face.Width / 2f, face.Y + face.Height / 2f);
            var decal = LastCentre.DistanceTo(centre);
            LastCentre = centre;
            return decal;
        }

        public double ProgressiveMean(IEnumerable<double> enu)
        {
            var arr = enu.ToArray();
            var coeffs = Enumerable.Range(0, arr.Length)
                .Select(x => Math.Sqrt((x + arr.Length / 100f) / (arr.Length + arr.Length / 100f))).ToArray();
            return arr.Zip(coeffs, (x, c) => x * c).Sum() / coeffs.Sum();
        }

        public Mat GetSubpicture(Rect rect, Mat inp = null)
        {
            return (inp ?? Input)[rect];
        }

        public double CalcMeanColor(Rect rect)
        {
            var slice = GetSubpicture(rect);
            InputBuffer.Add(slice);

            if (Denoise)
            {
                Cv2.FastNlMeansDenoisingColoredMulti(
                    InputBuffer, 
                    slice, 
                    InputBuffer.Count / 2, 
                    InputBuffer.Count,
                    searchWindowSize: 35);
            }

            var weights = new[] {0, 0.85, 0.4};

            return weights.Select((weight, c) => weight * slice.ExtractChannel(c).Mean().Val0).Sum() / weights.Sum();
        }

        public Rect GetSlice(double rx, double ry, double rw, double rh)
        {
            var fix_x = Face.Width * rx;
            var fix_y = Face.Height * ry;
            var fix_w = Face.Width * rw;
            var fix_h = Face.Height * rh;

            return new Rect(
                (int)(Face.X + fix_x - fix_w / 2),
                (int)(Face.Y + fix_y - fix_h / 2),
                (int)fix_w,
                (int)fix_h
                );
        }

        public void DrawAt(Mat img, Rect rect)
        {
            Output[rect] = img;
        }

        public void Execute(TextDlg text)
        {
            Output = Input.Clone();

            InputBuffer.Add(Input);
            BufferState++;
            BufferState %= BufferSize;
            TimesBuffer.Add(DateTime.Now - StartTime);

            if (!LockFace)
            {
                var input_g = new Mat();
                Cv2.CvtColor(Input, input_g, ColorConversionCodes.BGR2GRAY);
                Cv2.EqualizeHist(input_g, input_g);

                var detect = Classifier.DetectMultiScale(input_g, minNeighbors: 4, minSize: new Size(50, 50),
                    flags: HaarDetectionType.ScaleImage);

                if (detect.Length > 0)
                {
                    var biggest = detect.OrderByDescending(r => r.Width * r.Height).First();

                    if (CalcShift(biggest) > SHIFT_THRESHOLD)
                    {
                        Face = biggest;
                        InputBuffer.Clear();
                    }
                }
            }

            if (Face == OpenCvSharp.Rect.Empty)
            {
                AverageColorBuffer.Add(0);
                return;
            }

            var forehead = GetSlice(0.5, 0.15, 0.38, 0.14);

            Rect(Face, BLUE);
            Rect(forehead, GREEN);

            if (Colorify)
            {
                // todo
            }

            var fh_average = CalcMeanColor(forehead);
            AverageColorBuffer.Add(fh_average);

            var num_samples = TimesBuffer.Count;

            if (num_samples > MIN_SAMPLE_COUNT)
            {
                var time_start = TimesBuffer[0];
                var time_end = TimesBuffer[TimesBuffer.Count - 1];

                LastFPS = FPS;
                var dt = time_end - time_start;
                if (dt != TimeSpan.Zero)
                {
                    var fps = num_samples / dt.TotalSeconds;
                    FPS = (int) (fps * FPS_SMOOTHING + LastFPS * (1 - FPS_SMOOTHING));
                }

                var linear = MathNet.Numerics.Generate.LinearSpaced(num_samples, time_start.TotalSeconds, time_end.TotalSeconds);

                var window = MathNet.Numerics.Window.Hamming(num_samples);

                var interp =
                    window.Zip(linear.Select(i => MathNet.Numerics.Interpolate.Linear(TimesBuffer.Select(x => x.TotalSeconds), AverageColorBuffer).Interpolate(i)), (a, b) => a - b).ToList();

                Deviation = interp.Select(x => x - ProgressiveMean(interp)).ToList();

                var fourier_raw = Deviation.Select(x => new Complex32((float)x, 0)).ToArray();

                MathNet.Numerics.IntegralTransforms.Fourier.Forward(fourier_raw);

                Fourier = fourier_raw.Select(x => Math.Abs(x.Real)).ToArray();

                Frequencies = Enumerable.Range(0, (int)Math.Ceiling(num_samples / 2d + 1)).Select(x => x * 60.0 * FPS / num_samples).ToArray();

                try
                {
                    var zip = Frequencies.Zip(Fourier, (a, b) => new {a, b})
                        .Where(x => x.a > BPM_LOW && x.b < BPM_HIGH).ToList();

                    Frequencies = zip.Select(x => x.a).ToArray();
                    Fourier = zip.Select(x => x.b).ToArray();

                    BPMPosition = Array.IndexOf(Fourier, Fourier.Max());
                    BPM = Frequencies[BPMPosition];

                    BPMBuffer.Add(BPM);

                    CorrectBPM = ProgressiveMean(BPMBuffer);
                    text($"d={Deviation.Last():F3}", 20, 120);
                }
                catch
                {
                    ;
                }

                if (FPS != 0)
                Gap = (BufferSize - num_samples) / FPS;
            }
        }
    }
}
