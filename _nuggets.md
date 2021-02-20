# Nuggets 1

I started following the tutorial

I cloned the repo, then I coped the code into my own project.

git clone https://github.com/tensorflow/examples.git

I copied the required ios folder into this repo for me to work on an morph

examples/lite/examples/object_detection/ios/

cd ios

```s
pod install
```

Result

```s
Ignoring clocale-0.0.4 because its extensions are not built. Try: gem pristine clocale --version 0.0.4
Analyzing dependencies
Downloading dependencies
Generating Pods project
Integrating client project
Pod installation complete! There is 1 dependency from the Podfile and 2 total pods installed
```

I will now use finder to locate the project and open the newly create xcode workspace.

This workspace was created by the pod install.

open ObjectDetection.xcworkspace

I then ensured that the project had all of the signing/capabilites set for my Apple Dev Account, and then I did a run build for running and we have build success.

====

11:11

I am now going to follow the tutorial after I have had a walk.

Note: You do not need to build the entire TensorFlow library to run the demo; the demo uses CocoaPods to download the TensorFlow Lite library.

If you have installed this pod before, use the following command:

```s
pod update
```

Result

```s
Ignoring clocale-0.0.4 because its extensions are not built. Try: gem pristine clocale --version 0.0.4
Update all pods
Updating local specs repositories
Analyzing dependencies
Downloading dependencies
Installing TensorFlowLiteC 2.4.0 (was 2.1.0)
Installing TensorFlowLiteSwift 2.4.0 (was 2.1.0)
Generating Pods project
Integrating client project
Pod installation complete! There is 1 dependency from the Podfile and 2 total pods installed.
```

The sample app is a camera app that continuously detects the objects (bounding boxes and labels) in the frames seen by your device's back camera, using a quantized MobileNet SSD model trained on the COCO dataset.

the model is located in Model folder and is called `detect.tflite`

A TensorFlow ML model that has been compressed for use on mobile and embedded devices.

TF Lite converter - TensorFlow Lite uses the optimized `FlatBuffer` format to represent graphs. 
Therefore, a TensorFlow model (protocol buffer) needs to be converted into a FlatBuffer file before deploying to clients.

TF Lite interpreter - A class that does the job of a tf.Session(), only for TF Lite models as opposed to regular TensorFlow models.

Some interesting concepts here that I will become more familiar with over time.

11:46

https://www.tensorflow.org/lite/guide/inference

The term inference refers to the process of executing a TensorFlow Lite model on-device in order to make predictions based on input data. To perform an inference with a TensorFlow Lite model, you must run it through an interpreter. 

The TensorFlow Lite interpreter is designed to be lean and fast. The interpreter uses a static graph ordering and a custom (less-dynamic) memory allocator to ensure minimal load, initialization, and execution latency.

Important concepts
TensorFlow Lite inference typically follows the following steps:

1. Loading a model

You must load the .tflite model into memory, which contains the model's execution graph.

2. Transforming data

Raw input data for the model generally does not match the input data format expected by the model. 
- For example, you might need to resize an image or change the image format to be compatible with the model.

3. Running inference

- This step involves using the TensorFlow Lite API to execute the model. 
- It involves a few steps such as building the interpreter, and allocating tensors, as described in the following sections.

4. Interpreting output

- When you receive results from the model inference, you must interpret the tensors in a meaningful way that's useful in your application.
- For example, a model might return only a list of probabilities. It's up to you to map the probabilities to relevant categories and present it to your end-user.


So for me I guess we will get back the propability of a face recognition using the ideas I learned in this articel:

https://medium.com/@estebanuri/real-time-face-recognition-with-android-tensorflow-lite-14e9c6cc53a5

My summar of what I need to do to acheive the `inference` steps above

Using MLKit and IOS Camera (Note I have done this on Android, but on IOS I am writing the code as the article abive showed me the Java code for android, but I am not just trying to copy the Java code, I am trying to understand the process required and write the swift code myself as best i can with my limited knowledge and experience)

1. The face is detected on the input image.
 - This is a face object, this I have = DONE
2. The image is warped using the detected landmarks to align the face (so that all cropped faces have the eyes in the same position).
3. The face is cropped, and properly resized to feed the recognition Deep Learning model. 
 - Also some image pre-processing operations are done in this step (e. g. normalizing and “whitening” the face)
4. The most “juicy part”, is the one depicted as “Deep Neural Network”. 
 - We are going to focus more on this step in this code!

12:09

I am slowly getting some intelligenc here!

The main idea is that the deep neural network DNN takes as input a face F and gives as output a D = 128 dimensions vector (of floats). This vector E is known as embeddings. This embeedings are created such as the similarity between the two faces F1 and F2 can be computed simply as the euclidean distance between the embeddings E1 and E2.

See images/equation1

We can now compare two faces F1 and F2, by computing its Similarity, and then check it against some threshold. If lower we can say that both faces are from the same person.

Deep theory:
https://learnopencv.com/face-recognition-an-introduction-for-beginners/

For more details, here is a great article [3] from Satya Mallick that explains more in detail the basics, how a new face is registered to the system, and introduces some important concepts like the triplet loss and the kNN algorithms.


## Adding the Face Recognition Step

Adding the Face Recognition Step

The original sample comes with other DL model and it computes the results in one single step. For this app, we need to implement several steps process. Most of the work will consist in splitting the detection, first the face detection and second to the face recognition. For the face detection step we are going to use the Google ML kit.

Let’s add the ML kit dependency to our project by adding the following line to the build.gradle file:

The original sample comes with other DL model and it computes the results in one single step. For this app, we need to implement several steps process. Most of the work will consist in splitting the detection, first the face detection and second to the face recognition. For the face detection step we are going to use the Google ML kit.

Let’s add the ML kit dependency to our project by adding the following line to the build.gradle file:

I will come back to this. I need to go back to the tutorial to complete it.

====

Re-reading the goal of th eproject we see that it was really intended to show how to create your own model using AutoML Vision Edge

In this tutorial you will download an exported custom TensorFlow Lite model from AutoML Vision Edge. You will then run a pre-made iOS app that uses the model to detect multiple objects within an image (with bounding boxes), and provide custom labeling of object categories.

the root how to guides from Google for this example project ie to create the model ceom from
https://cloud.google.com/vision/automl/object-detection/docs/how-to

Key steps to entire concept of creating a model:


- Before you begin
Set up your Google Cloud Platform project, authentication, and enable AutoML Vision Object Detection.

- Preparing your training data
Learn best practices in organizing and annotating the images you will use to train your model.

- Formatting a training data CSV
Create a correctly formatted CSV file for the data set used to train your model.

- Creating datasets and importing images
Create the dataset and import the training data used to train your model.

- Annotating imported training images
Add, delete, or modify bounding box and label annotations for imported training images.

- Training Cloud-hosted models
Train your custom model and get the status of the training operation.

- Training Edge (exportable) models
Train your custom exportable Edge model and get the status of the training operation.

- Evaluating models
Review the performance of your model.

- Deploying models
Deploy your model for use after training completes.

- Making individual predictions
Use your custom model to annotate an individual prediction image with labels and bounding boxes online.

- Making batch predictions
Use your custom model to annotate a batch of prediction images with labels and bounding boxes online.

- Exporting Edge models
Export your different trained Edge model formats to Google Cloud Storage and for use on edge devices.

- Undeploying models
Undeploy your model after you are done using them to avoid further hosting charges.

- Managing datasets
Manage datasets associated with your project.

- Managing models
Manage your custom models.

- Working with long-running operations
Get the status of long-running operations.

- Base64 encode
Use native "base64" utilities to encode a binary image into ASCII text data to send in an API request.

- Filtering when listing
Learn how to filter results when listing resources, operations, and metrics.

The cool thing here is I can now write a course on how to train your own model vs using other industry standard models. 
I could literally dedicate myself to Computer.Vision

===

I deas for domains:

noisiv.vision

====

Back to the article:

ModelDataHandler.swift

The first block of interest (after the necessary imports) is property declarations. 

The tfLite model inputShape parameters (batchSize, inputChannels, inputWidth, inputHeight) can be found in tflite_metadata.json you will have this file when exporting the tflite model. For more info visit the Exporting Edge models how-to topic.

The example of tflite_metadata.json looks similar to the following code:

Note I am not using the exported json, it is pasted here to remind me how I would understand what Constants maybe needed?

```json
{
    "inferenceType": "QUANTIZED_UINT8",
    "inputShape": [
        1,   // This represents batch size
        512,  // This represents image width
        512,  // This represents image Height
        3  //This represents inputChannels
    ],
    "inputTensor": "normalized_input_image_tensor",
    "maxDetections": 20,  // This represents max number of boxes.
    "outputTensorRepresentation": [
        "bounding_boxes",
        "class_labels",
        "class_confidences",
        "num_of_boxes"
    ],
    "outputTensors": [
        "TFLite_Detection_PostProcess",
        "TFLite_Detection_PostProcess:1",
        "TFLite_Detection_PostProcess:2",
        "TFLite_Detection_PostProcess:3"
    ]
}
...
```


Note: These parameters will be used to calculate the byte size of the image by using the following formula: byteCount: batchSize * inputWidth * inputHeight * inputChannels,

let batchSize = 1 //Number of images to get prediction, the model takes 1 image at a time
let inputChannels = 3 //The pixels of the image input represented in RGB values
let inputWidth = 300 //Width of the image
let inputHeight = 300 //Height of the image

The init method, which creates the Interpreter with Model path and InterpreterOptions, then allocates memory for the model's input.


```swift
// MARK: - Initialization

  /// A failable initializer for `ModelDataHandler`. A new instance is created if the model and
  /// labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
  init?(modelFileInfo: FileInfo, labelsFileInfo: FileInfo, threadCount: Int = 1) {
    let modelFilename = modelFileInfo.name

    // Construct the path to the model file.
    guard let modelPath = Bundle.main.path(
      forResource: modelFilename,
      ofType: modelFileInfo.extension
    ) else {
      print("Failed to load the model file with name: \(modelFilename).")
      return nil
    }

    // Specify the options for the `Interpreter`.
    self.threadCount = threadCount
    var options = Interpreter.Options()
    options.threadCount = threadCount
    do {
      // Create the `Interpreter`.
      interpreter = try Interpreter(modelPath: modelPath, options: options)
      // Allocate memory for the model's input `Tensor`s.
      try interpreter.allocateTensors()
    } catch let error {
      print("Failed to create the interpreter with error: \(error.localizedDescription)")
      return nil
    }

    super.init()

    // Load the classes listed in the labels file.
    loadLabels(fileInfo: labelsFileInfo)
  }
```

Swift's guard keyword lets us check an optional exists and exit the current scope if it doesn't, which makes it perfect for early returns in methods

https://learnappmaking.com/swift-guard-let-statement-how-to/

Undestanding running the model

The below runModel method:

- Scales input image to aspect ratio for which model is trained.
- Removes the alpha component from the image buffer to get the RGB data.
- Copies the RGB data to the input Tensor.
- Runs inference by invoking the Interpreter.
- Gets the output from the Interpreter.
- Formats output.

```swift
/// This class handles all data preprocessing and makes calls to run inference on a given frame
  /// through the `Interpreter`. It then formats the inferences obtained and returns the top N
  /// results for a successful inference.
  func runModel(onFrame pixelBuffer: CVPixelBuffer) -> Result? {
    let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
    let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
    let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
    assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
             sourcePixelFormat == kCVPixelFormatType_32BGRA ||
               sourcePixelFormat == kCVPixelFormatType_32RGBA)


    let imageChannels = 4
    assert(imageChannels >= inputChannels)

    // Crops the image to the biggest square in the center and scales it down to model dimensions.
    let scaledSize = CGSize(width: inputWidth, height: inputHeight)
    guard let scaledPixelBuffer = pixelBuffer.resized(to: scaledSize) else {
      return nil
    }

    let interval: TimeInterval
    let outputBoundingBox: Tensor
    let outputClasses: Tensor
    let outputScores: Tensor
    let outputCount: Tensor
    do {
      let inputTensor = try interpreter.input(at: 0)

      // Remove the alpha component from the image buffer to get the RGB data.
      guard let rgbData = rgbDataFromBuffer(
        scaledPixelBuffer,
        byteCount: batchSize * inputWidth * inputHeight * inputChannels,
        isModelQuantized: inputTensor.dataType == .uInt8
      ) else {
        print("Failed to convert the image buffer to RGB data.")
        return nil
      }

      // Copy the RGB data to the input `Tensor`.
      try interpreter.copy(rgbData, toInputAt: 0)

      // Run inference by invoking the `Interpreter`.
      let startDate = Date()
      try interpreter.invoke()
      interval = Date().timeIntervalSince(startDate) * 1000

      outputBoundingBox = try interpreter.output(at: 0)
      outputClasses = try interpreter.output(at: 1)
      outputScores = try interpreter.output(at: 2)
      outputCount = try interpreter.output(at: 3)
    } catch let error {
      print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
      return nil
    }

    // Formats the results
    let resultArray = formatResults(
      boundingBox: [Float](unsafeData: outputBoundingBox.data) ?? [],
      outputClasses: [Float](unsafeData: outputClasses.data) ?? [],
      outputScores: [Float](unsafeData: outputScores.data) ?? [],
      outputCount: Int(([Float](unsafeData: outputCount.data) ?? [0])[0]),
      width: CGFloat(imageWidth),
      height: CGFloat(imageHeight)
    )

    // Returns the inference time and inferences
    let result = Result(inferenceTime: interval, inferences: resultArray)
    return result
  }
```

Crops the image to the biggest square in the center and scales it down to model dimensions:
```swift
// Crops the image to the biggest square in the center and scales it down to model dimensions.
    let scaledSize = CGSize(width: inputWidth, height: inputHeight)
    guard let scaledPixelBuffer = pixelBuffer.resized(to: scaledSize) else {
      return nil
    }
```
Remove the alpha component from the image buffer to get the RGB data:
```swift
 // Remove the alpha component from the image buffer to get the RGB data.
      guard let rgbData = rgbDataFromBuffer(
        scaledPixelBuffer,
        byteCount: batchSize * inputWidth * inputHeight * inputChannels,
        isModelQuantized: inputTensor.dataType == .uInt8
      ) else {
        print("Failed to convert the image buffer to RGB data.")
        return nil
      }
```

Copy the RGB data to the input `Tensor`.
```swift
try interpreter.copy(rgbData, toInputAt: 0)
```

Run inference by invoking the Interpreter:

Formats the results
```swift
    let resultArray = formatResults(
      boundingBox: [Float](unsafeData: outputBoundingBox.data) ?? [],
      outputClasses: [Float](unsafeData: outputClasses.data) ?? [],
      outputScores: [Float](unsafeData: outputScores.data) ?? [],
      outputCount: Int(([Float](unsafeData: outputCount.data) ?? [0])[0]),
      width: CGFloat(imageWidth),
      height: CGFloat(imageHeight)
    )

    // Returns the inference time and inferences
    let result = Result(inferenceTime: interval, inferences: resultArray)
    return result
```

`func formatResults(boundingBox: [Float], outputClasses: [Float], outputScores: [Float], outputCount: Int, width: CGFloat, height: CGFloat) -> [Inference]{`

```swift
/// Filters out all the results with confidence score < threshold and returns the top N results
  /// sorted in descending order.
  func formatResults(boundingBox: [Float], outputClasses: [Float], outputScores: [Float], outputCount: Int, width: CGFloat, height: CGFloat) -> [Inference]{
    var resultsArray: [Inference] = []
    if (outputCount == 0) {
      return resultsArray
    }
    for i in 0...outputCount - 1 {

      let score = outputScores[i]

      // Filters results with confidence < threshold.
      guard score >= threshold else {
        continue
      }

      // Gets the output class names for detected classes from labels list.
      let outputClassIndex = Int(outputClasses[i])
      let outputClass = labels[outputClassIndex + 1]

      var rect: CGRect = CGRect.zero

      // Translates the detected bounding box to CGRect.
      rect.origin.y = CGFloat(boundingBox[4*i])
      rect.origin.x = CGFloat(boundingBox[4*i+1])
      rect.size.height = CGFloat(boundingBox[4*i+2]) - rect.origin.y
      rect.size.width = CGFloat(boundingBox[4*i+3]) - rect.origin.x

      // The detected corners are for model dimensions. So we scale the rect with respect to the
      // actual image dimensions.
      let newRect = rect.applying(CGAffineTransform(scaleX: width, y: height))

      // Gets the color assigned for the class
      let colorToAssign = colorForClass(withIndex: outputClassIndex + 1)
      let inference = Inference(confidence: score,
                                className: outputClass,
                                rect: newRect,
                                displayColor: colorToAssign)
      resultsArray.append(inference)
    }

    // Sort results in descending order of confidence.
    resultsArray.sort { (first, second) -> Bool in
      return first.confidence  > second.confidence
    }

    return resultsArray
  }
```

Explaination

Filters out all the results with confidence score < threshold and returns the top N results sorted in descending order:

```swift
func formatResults(boundingBox: [Float], outputClasses: [Float],
  outputScores: [Float], outputCount: Int, width: CGFloat, height: CGFloat)
  -> [Inference]{
    var resultsArray: [Inference] = []
    for i in 0...outputCount - 1 {

      let score = outputScores[i]
```

Filters results with confidence < threshold:

```swift
      guard score >= threshold else {
        continue
      }
```

Gets the output class names for detected classes from labels list:

```swift
      let outputClassIndex = Int(outputClasses[i])
      let outputClass = labels[outputClassIndex + 1]

      var rect: CGRect = CGRect.zero
```

Translates the detected bounding box to CGRect. 

__Note:__ I should draw this out sometime to show what it is doing in a pictorial way.

```swift
rect.origin.y = CGFloat(boundingBox[4*i])
rect.origin.x = CGFloat(boundingBox[4*i+1])
rect.size.height = CGFloat(boundingBox[4*i+2]) - rect.origin.y
rect.size.width = CGFloat(boundingBox[4*i+3]) - rect.origin.x
```

The detected corners are for model dimensions. So we scale the rect with respect to the actual image dimensions.

```swift
let newRect = rect.applying(CGAffineTransform(scaleX: width, y: height))
```

Gets the color assigned for the class:
```swift

let colorToAssign = colorForClass(withIndex: outputClassIndex + 1)
      let inference = Inference(confidence: score,
                                className: outputClass,
                                rect: newRect,
                                displayColor: colorToAssign)
      resultsArray.append(inference)
    }

    // Sort results in descending order of confidence.
    resultsArray.sort { (first, second) -> Bool in
      return first.confidence  > second.confidence
    }

    return resultsArray
  }
```


https://cloud.google.com/vision/automl/object-detection/docs/tflite-ios-tutorial

Learn more about TFLite from the official documentation and the code repository. : `https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite`
Try some other TFLite ready models including a speech hot-word detector and an on-device version of smart-reply.
Learn more about TensorFlow in general with TensorFlow's getting started documentation.

====

It is now 13:57, I have also bakded some grandad cookies, going o get out of oven now.



