using System.Collections;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using Vuforia;
using UnityEngine.EventSystems;


public class main : MonoBehaviour
{
    public GUIStyle custom;

    private PIXEL_FORMAT mPixelFormat = PIXEL_FORMAT.UNKNOWN_FORMAT;

    private bool mAccessCameraImage = true;
    private bool mFormatRegistered; // false
    private bool debugDisplay; // false
    private bool guiEnabled = true;

    private Texture2D tex;
    private Color32[] pixel32;

    private GCHandle pixelHandle;
    private IntPtr pixelPtr;

    public GameObject ground;
    public GameObject display;

    private byte[] pixels;
    private int width;
    private int height;
    private int rowsToCrop;

    // Start is called before the first frame update
    void Start()
    {
#if UNITY_EDITOR
        mPixelFormat = PIXEL_FORMAT.RGBA8888; // Need Grayscale for Editor
#else
        mPixelFormat = PIXEL_FORMAT.RGB888; // Use RGB888 for mobile
#endif

        // Register Vuforia life-cycle callbacks:
        VuforiaARController.Instance.RegisterVuforiaStartedCallback(OnVuforiaStarted);
        VuforiaARController.Instance.RegisterOnPauseCallback(OnPause);

        InitTexture();
        display.GetComponent<Renderer>().material.mainTexture = tex;
    }

    // Update is called once per frame
    void Update()
    {
        if (mFormatRegistered)
        {
            if (mAccessCameraImage)
            {
                Vuforia.Image image = CameraDevice.Instance.GetCameraImage(mPixelFormat);

                if (image != null && image.Height != 0)
                {
                    pixels = image.Pixels;
                    width = image.Width;
                    height = image.Height;
                    /*
                    float screenRatio = Screen.width / (float) Screen.height;
                    float imageRatio = width / (float) height;
                    float percentage = screenRatio / (imageRatio / 100);
                    int reduction = (int)Math.Ceiling(height * percentage);

                    Debug.Log("screen: "+ screenRatio + ", image: " + imageRatio);
                    Debug.Log("percentage: "+ percentage + ", reduction: " + reduction);
                    rowsToCrop = (reduction / 2);
                    */

                }
            }
        }

        if (Input.GetMouseButtonDown(0))
        {
            //Debug.Log(Input.mousePosition);
            //Debug.Log("Image: " + width + ", " + height);
            //Debug.Log("Screen: " + Screen.width + ", " + Screen.height);
            //guiEnabled = !guiEnabled;
        }

        MatToTexture2D();
    }

    void OnVuforiaStarted()
    {

        // Try register camera image format
        if (CameraDevice.Instance.SetFrameFormat(mPixelFormat, true))
        {
            Debug.Log("Successfully registered pixel format " + mPixelFormat.ToString());

            mFormatRegistered = true;

            Vuforia.Image image = CameraDevice.Instance.GetCameraImage(mPixelFormat);
        }
        else
        {
            Debug.LogError(
                "\nFailed to register pixel format: " + mPixelFormat.ToString() +
                "\nThe format may be unsupported by your device." +
                "\nConsider using a different pixel format.\n");

            mFormatRegistered = false;
        }

    }

    // Called when app is paused / resumed
    void OnPause(bool paused)
    {
        if (paused)
        {
            Debug.Log("App was paused");
            UnregisterFormat();
        }
        else
        {
            Debug.Log("App was resumed");
            RegisterFormat();
        }
    }

    void RegisterFormat()
    {
        if (CameraDevice.Instance.SetFrameFormat(mPixelFormat, true))
        {
            Debug.Log("Successfully registered camera pixel format " + mPixelFormat.ToString());
            mFormatRegistered = true;
        }
        else
        {
            Debug.LogError("Failed to register camera pixel format " + mPixelFormat.ToString());
            mFormatRegistered = false;
        }
    }

    void UnregisterFormat()
    {
        Debug.Log("Unregistering camera pixel format " + mPixelFormat.ToString());
        CameraDevice.Instance.SetFrameFormat(mPixelFormat, false);
        mFormatRegistered = false;
    }



    /////// THA DEBUG ZONE /////////////


    void InitTexture()
    {
        Vuforia.Image image = CameraDevice.Instance.GetCameraImage(mPixelFormat);
        tex = new Texture2D(1920, 1080, TextureFormat.ARGB32, false);
        pixel32 = tex.GetPixels32();
        //Pin pixel32 array
        pixelHandle = GCHandle.Alloc(pixel32, GCHandleType.Pinned);
        //Get the pinned address
        pixelPtr = pixelHandle.AddrOfPinnedObject();
        Debug.Log("Tex Initialised");
    }

    void MatToTexture2D()
    {
        //Convert Mat to Texture2D
        int tw = tex.width;
        int th = tex.height;

        OpenCVInterop.GetRawImageBytes(ref pixelPtr, ref tw, ref th);
        //Update the Texture2D with array updated in C++
        tex.SetPixels32(pixel32);
        tex.Apply();
        //Debug.Log("Tex Updated");
    }

    void OnApplicationQuit()
    {
        //Free handle
        pixelHandle.Free();
    }

        void OnGUI()
        {
        if (guiEnabled)
        {
            if (debugDisplay)
            {
                GUI.DrawTexture(new Rect(0, 0, Screen.width, Screen.height), tex, ScaleMode.StretchToFill, false);
            }
            if (GUI.Button(new Rect(Screen.width - 400, Screen.height / 2 - 100, 400, 200), "Align Content"))
            {
                if (OpenCVInterop.sendImage(ref pixels, ref width, ref height, ref rowsToCrop) == 1)
                {
                    ground.transform.position = new Vector3(0, 0, 0);
                    ground.transform.rotation = new Quaternion(0, 0, 0, 0);
                    Vector3 offset = ground.transform.localPosition;

                    /*Vector2[] points = { new Vector2(0, 0),
                                         new Vector2(0, 1440),
                                         new Vector2(800, 0),
                                         new Vector2(800, 1440),
                                       };*/


                    Vector2[] points = { new Vector2(0+150, 0+150),
                                                 new Vector2(0+150, 1440/4 +150),
                                                 new Vector2(800/4 +150, 0+150),
                                                 new Vector2(800/4 +150, 1440/4 +150),
                                               };


                    Vector3[] vertices = new Vector3[4];

                    for (int i = 0; i < 4; i++) vertices[i] = ground.transform.TransformPoint(ground.GetComponent<MeshFilter>().mesh.vertices[i]); //Get world positions of vertices
                    Debug.Log("Vertices: " + vertices[0] + ", " + vertices[1] + ", " + vertices[2] + ", " + vertices[3]);

                    float[] rotationMatrix = new float[9];
                    float[] translationMatrix = new float[3];
                    Matrix4x4 transformationMatrix = new Matrix4x4();

                    int h = Screen.height;
                    int w = Screen.width;

                    //OpenCVInterop.ComputePNP(ref vertices, ref points, ref rotationMatrix, ref translationMatrix);

                    IntPtr rotationPtr, translationPtr;

                    Debug.Log("Rotation Matrix: " + rotationMatrix[0] + ", " + rotationMatrix[1] + ", " + rotationMatrix[2]);
                    Debug.Log("Size of float: " + sizeof(float));

                    OpenCVInterop.ComputePNP(ref vertices, ref points, out rotationPtr, out translationPtr, ref w, ref h);
                    Debug.Log("AFTER DLL CALL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~");

                    Marshal.Copy(rotationPtr, rotationMatrix, 0, rotationMatrix.Length);
                    Marshal.Copy(translationPtr, translationMatrix, 0, translationMatrix.Length);

                    Marshal.FreeCoTaskMem(rotationPtr);
                    Marshal.FreeCoTaskMem(translationPtr);


                    ////////////////////////////

                    transformationMatrix[0, 0] = rotationMatrix[0];
                    transformationMatrix[0, 1] = rotationMatrix[1];
                    transformationMatrix[0, 2] = rotationMatrix[2];
                    transformationMatrix[0, 3] = translationMatrix[0];

                    transformationMatrix[1, 0] = rotationMatrix[3];
                    transformationMatrix[1, 1] = rotationMatrix[4];
                    transformationMatrix[1, 2] = rotationMatrix[5];
                    transformationMatrix[1, 3] = translationMatrix[1];

                    transformationMatrix[2, 0] = rotationMatrix[6];
                    transformationMatrix[2, 1] = rotationMatrix[7];
                    transformationMatrix[2, 2] = rotationMatrix[8];
                    transformationMatrix[2, 3] = translationMatrix[2];

                    transformationMatrix[3, 3] = 1;

                    //Convert from OpenCV to Unity coordinates
                    Matrix4x4 invertYM = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(1, -1, 1));

                    transformationMatrix = transformationMatrix * invertYM;
                    //transformationMatrix = transformationMatrix * this.gameObject.transform.localToWorldMatrix;
                    Debug.Log(transformationMatrix.ToString());



                    Vector3 position = Vector3.zero;
                    Quaternion rotation = Quaternion.identity;

                    position = transformationMatrix.MultiplyPoint3x4(ground.transform.position);
                    rotation = ground.transform.rotation * Quaternion.LookRotation(transformationMatrix.GetColumn(2), -transformationMatrix.GetColumn(1));

                    var p = ground.transform.parent;

                    //put stadium into camera's child
                    ground.transform.parent = this.gameObject.transform;

                    //apply stored transformation to recreate the position of stadium when camera is at (0,0,0) to localPosition (in relative to camera movement)
                    ground.transform.localPosition = position;
                    ground.transform.localRotation = rotation;

                    //set back the parent of the stadium to as before (highest)
                    ground.transform.parent = p;

                    //different coordinate system therefore inverse y, use translate instead of just transform.position as we are moving along self coordinate system
                    ground.transform.Translate(-(new Vector3(offset.x, -offset.y, offset.z)), Space.Self);

                    /*
                    ground.transform.position = transformationMatrix.MultiplyPoint3x4(ground.transform.position);
                    ground.transform.rotation *= Quaternion.LookRotation(transformationMatrix.GetColumn(2), -transformationMatrix.GetColumn(1));
                    */
                    //ground.GetComponent<MeshRenderer>().enabled = true;

                }
            }

            if (GUI.Button(new Rect(0, 0, 300, 200), "Toggle Rendering")) // Toggle rendering of AR content
            {
                ground.SetActive(!ground.activeSelf);
            }

            if (GUI.Button(new Rect(0, 205, 300, 200), "Debug Display")) // Toggle fullscreen openCV output
            {
                debugDisplay = !debugDisplay;
            }

        }
    }


}

internal static class OpenCVInterop
{
    [DllImport("__Internal")]
    internal static extern int sendImage(ref byte[] image, ref int width, ref int height, ref int crop);

    [DllImport("__Internal")]
    internal static extern void ComputePNP(ref Vector3[] op, ref Vector2[] ip, out IntPtr rvPtr, out IntPtr tvPtr, ref int width, ref int height);

    [DllImport("__Internal")]
    internal static extern void GetRawImageBytes(ref IntPtr data, ref int width, ref int height);
}