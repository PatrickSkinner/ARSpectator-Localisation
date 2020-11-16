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

    private bool readyToClick = false;
    private bool isDone = false;
    private String isLocalised = "Localise";

    private Texture2D tex;
    private Color32[] pixel32;

    private GCHandle pixelHandle;
    private IntPtr pixelPtr;

    public GameObject ground;
    public GameObject display;
    public GameObject stadium;

    private byte[] pixels;
    private int width;
    private int height;
    private int rowsToCrop;

    private bool debugMode = false;
    // Scalar(35, 30, 0), Scalar(80, 219, 197)
    String hL = "35";
    String hU = "80";
    String sL = "30";
    String sU = "219";
    String vL = "0";
    String vU = "197";

    String rho = "2";
    String vote = "150";
    String minLen = "275";
    String focalLength = "1410";

    bool toggleBool = false;

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
            //Debug.Log("Click");
            
            OpenCVInterop.updateThreshold(int.Parse(hL), int.Parse(hU), int.Parse(sL), int.Parse(sU), int.Parse(vL), int.Parse(vU));
            OpenCVInterop.updateHough(int.Parse(rho), int.Parse(vote), int.Parse(minLen), int.Parse(focalLength), toggleBool );
            OpenCVInterop.sendClick((int) Input.mousePosition.x,(int) Input.mousePosition.y);
            if (readyToClick)
            {
                localise();
            }
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

    void DoTheThing()
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
        //Debug.Log("Vertices: " + vertices[0] + ", " + vertices[1] + ", " + vertices[2] + ", " + vertices[3]);

        float[] rotationMatrix = new float[9];
        float[] translationMatrix = new float[3];
        Matrix4x4 transformationMatrix = new Matrix4x4();

        int h = Screen.height;
        int w = Screen.width;

        //OpenCVInterop.ComputePNP(ref vertices, ref points, ref rotationMatrix, ref translationMatrix);

        IntPtr rotationPtr, translationPtr;

        //Debug.Log("Rotation Matrix: " + rotationMatrix[0] + ", " + rotationMatrix[1] + ", " + rotationMatrix[2]);
        //Debug.Log("Size of float: " + sizeof(float));

        OpenCVInterop.ComputePNP(ref vertices, ref points, out rotationPtr, out translationPtr, ref w, ref h);
        //Debug.Log("AFTER DLL CALL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~");

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
        Matrix4x4 invertYM = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(-1, 1, -1));

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

    void OnApplicationQuit()
    {
        //Free handle
        pixelHandle.Free();
    }

    void OnGUI()
    {
        if (GUI.Button(new Rect(0, Screen.height - 100, 300, 100), "Toggle Debug Mode")) // Toggle debug mode
        {
            debugMode = (!debugMode);
        }
        if (!debugMode)
        {
            if (guiEnabled)
            {


                
                /*
                if (GUI.Button(new Rect(Screen.width - 400, Screen.height / 2 - 100, 400, 200), "Align Content"))
                {
                    int flag = 0;

                    flag = OpenCVInterop.sendImage(ref pixels, ref width, ref height, ref rowsToCrop);
                

                    if (flag == 1)
                    {
                        DoTheThing();
                    }
                }*/
                if (GUI.Button(new Rect(Screen.width / 2 - 150, Screen.height - 100, 300, 100), isLocalised))
                {
                    readyToClick = true;
                    isLocalised = "Select a Line";
                }
                /*
                if (GUI.Button(new Rect(0, 0, 300, 200), "Toggle Rendering")) // Toggle rendering of AR content
                {
                    ground.SetActive(!ground.activeSelf);
                }
                */

            }
        } else { // DEBUG MODE ENABLE, SHOW DEBUG OPTIONS;
            hL = GUI.TextArea(new Rect(25, 1080/4, 200, 60), hL);
            hU = GUI.TextArea(new Rect(25, 1080 / 4 +(60*1), 200, 60), hU);
            sL = GUI.TextArea(new Rect(25, 1080 / 4 + (60 * 2), 200, 60), sL);
            sU = GUI.TextArea(new Rect(25, 1080 / 4 + (60 * 3), 200, 60), sU);
            vL = GUI.TextArea(new Rect(25, 1080 / 4 + (60 * 4), 200, 60), vL);
            vU = GUI.TextArea(new Rect(25, 1080 / 4 + (60 * 5), 200, 60), vU);

            rho = GUI.TextArea(new Rect(1920-500, 1080 / 4, 200, 60), rho);
            vote = GUI.TextArea(new Rect(1920 - 500, 1080 / 4 + (60 * 1), 200, 60), vote);
            minLen = GUI.TextArea(new Rect(1920 - 500, 1080 / 4 + (60 * 2), 200, 60), minLen);

            focalLength = GUI.TextArea(new Rect(1920 - 500, 1080 / 4 + (60 * 5), 200, 60), focalLength);

            if (GUI.Button(new Rect(0, 0, 300, 200), "Debug Display")) // Toggle fullscreen openCV output
            {
                debugDisplay = !debugDisplay;
            }

            if (GUI.Button(new Rect(1920-500, 1080-400, 300, 200), "Toggle Stadium")) // Toggle rendering of AR content
            {
                stadium.SetActive(!stadium.activeSelf);
            }
            if (GUI.Button(new Rect(1920 - 500-300, 1080 - 400, 300, 200), "Toggle Pitch")) // Toggle rendering of AR content
            {
                ground.SetActive(!ground.activeSelf);
            }

            toggleBool = GUI.Toggle(new Rect(Screen.width - 200, 50, 100, 30), toggleBool, "Toggle");

            if (debugDisplay)
            {
                GUI.DrawTexture(new Rect(0, 0, Screen.width, Screen.height), tex, ScaleMode.StretchToFill, false);
            }
        }
    }






    private bool localise()
    {
        var watch = new System.Diagnostics.Stopwatch();

        watch.Start();

        ground.transform.position = new Vector3(0, 0, 0);
        ground.transform.rotation = new Quaternion(0, 0, 0, 0);

        
        int selectedLine = 0;
        /*
        if (currentLine.Equals("Left"))
        {
            selectedLine = 0;
        }
        else if (currentLine.Equals("Center"))
        {
            selectedLine = 1;
        }
        else
        {
            selectedLine = 2;
        }
        */

        if(selectedLine == 0)
        {
            //ground.transform.rotation.SetEulerAngles(0, 0, 0);
        }
        if(selectedLine == 2)
        {
            //ground.transform.rotation.SetEulerAngles(0, 0, (float) - Math.PI/2);
        }

        if (OpenCVInterop.sendImage(ref pixels, ref width, ref height, ref selectedLine) == 1)
        {
            Vector3 offset = ground.transform.localPosition;


            // THIS DOESNT DO ANYTHING ATM, LEGACY CODE, REFACTOR LATER
            Vector2[] points = { new Vector2(0+150, 0+150),
                                 new Vector2(0+150, 1440/4 +150),
                                 new Vector2(800/4 +150, 0+150),
                                 new Vector2(800/4 +150, 1440/4 +150),
                               };


            Vector3[] vertices = new Vector3[4];

            if (selectedLine == 2)
            {
                Vector3[] tempVert = new Vector3[4];
                tempVert[0] = vertices[3];
                tempVert[1] = vertices[2];
                tempVert[2] = vertices[1];
                tempVert[3] = vertices[0];
                vertices = tempVert;
            }

            

            for (int i = 0; i < 4; i++) vertices[i] = ground.transform.TransformPoint(ground.GetComponent<MeshFilter>().mesh.vertices[i]); //Get world positions of vertices

            Matrix4x4 transformationMatrix = OpenCVInterop.ComputePNP(points, vertices);

            Vector3 internalPosition = Vector3.zero;
            Quaternion internalRotation = Quaternion.identity;

            internalPosition = transformationMatrix.MultiplyPoint3x4(ground.transform.position);
            internalRotation = ground.transform.rotation * Quaternion.LookRotation(transformationMatrix.GetColumn(2), -transformationMatrix.GetColumn(1));


            var p = ground.transform.parent;

            //make stadium the camera's child
            ground.transform.parent = this.gameObject.transform;

            //apply stored transformation to recreate the position of stadium when camera is at (0,0,0) to localPosition (in relative to camera movement)
            ground.transform.localPosition = internalPosition;
            ground.transform.localRotation = internalRotation;

            //set back the parent of the stadium to what it was before (highest)
            ground.transform.parent = p;

            //different coordinate system therefore inverse y, use translate instead of just transform.position as we are moving along self coordinate system
            ground.transform.Translate(-(new Vector3(offset.x, -offset.y, offset.z)), Space.Self);

            isDone = true;
            isLocalised = "Localisation Complete";
            watch.Stop();

            Debug.Log("Execution Time: " + watch.ElapsedMilliseconds + " ms");
            readyToClick = false;
            MatToTexture2D();
            return true; // Success
        }
        else
        {
            isLocalised = "Localisation Failed";
            readyToClick = false;
            MatToTexture2D();
            return false; // Failure 
        }
    }

}

internal static class OpenCVInterop
{
    [DllImport("__Internal")]
    internal static extern int sendImage(ref byte[] image, ref int width, ref int height, ref int crop);

    [DllImport("__Internal")]
    internal static extern int sendClick(int x, int y);

    [DllImport("__Internal")]
    internal static extern void updateThreshold(int hl, int hu, int sl, int su, int vl, int vu);

    [DllImport("__Internal")]
    internal static extern void updateHough(int inRho, int inVote, int inMinLen, int focalLen, bool toggle);

    [DllImport("__Internal")]
    internal static extern void ComputePNP(ref Vector3[] op, ref Vector2[] ip, out IntPtr rvPtr, out IntPtr tvPtr, ref int width, ref int height);

    [DllImport("__Internal")]
    internal static extern void GetRawImageBytes(ref IntPtr data, ref int width, ref int height);

    internal static Matrix4x4 ComputePNP(Vector2[] points2d, Vector3[] points3d)
    {
        float[] rotationMatrix = new float[9];
        float[] translationMatrix = new float[3];
        Matrix4x4 transformationMatrix = new Matrix4x4();

        IntPtr rotationPtr, translationPtr;
        int h = Screen.height;
        int w = Screen.width;
        ComputePNP(ref points3d, ref points2d, out rotationPtr, out translationPtr, ref w, ref h);
        Marshal.Copy(rotationPtr, rotationMatrix, 0, rotationMatrix.Length);
        Marshal.Copy(translationPtr, translationMatrix, 0, translationMatrix.Length);

        Marshal.FreeCoTaskMem(rotationPtr);
        Marshal.FreeCoTaskMem(translationPtr);

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
        //Matrix4x4 invertYM = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(1, -1, 1));
        Matrix4x4 invertYM = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(-1, 1, -1));

        transformationMatrix = transformationMatrix * invertYM;

        return transformationMatrix;
    }
}
