//
//  ContentView.swift
//  SwiftMnist
//
//  Created by Miguel de Icaza on 6/1/20.
//  Copyright © 2020 Miguel de Icaza. All rights reserved.
//

import SwiftUI
import CoreGraphics


let factor:CGFloat = 6.0

struct Drawing {
    var points: [CGPoint] = [CGPoint]()
}

struct DrawingPad: View {
    @Binding var currentDrawing: Drawing
    @Binding var drawings: [Drawing]
    @Binding var guesses: [Float]
    @Binding var result: Int
    
    func updateGuess ()
    {
        // Render the vector to a bitmap of ours
        let ctx = CGContext.init(data: nil, width: 28, height: 28, bitsPerComponent: 8, bytesPerRow: 28*4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue)!
        ctx.translateBy(x: 0, y: 28)
        ctx.scaleBy(x: 1, y: -1)
        
        ctx.setStrokeColor(NSColor.black.cgColor)
        for drawing in self.drawings {
            if drawing.points.count < 2 {
                continue
            }
    
            let mapped = drawing.points.map { CGPoint (x: $0.x/factor, y: $0.y/factor) }
            ctx.beginPath()
            ctx.move(to: mapped [0])
            for i in 1..<mapped.count {
                ctx.addLine(to: mapped [i])
            }
            ctx.strokePath()
        }
        // Turn bitmap into float array
        var size: Int = 0
        let mnist_input = mnist_get_input_image(mnist, &size)!
        let ctxdata = ctx.data!.bindMemory(to: Int32.self, capacity: 28*28*4)
        var midx = 0
        print ("-------------------")
        for _ in 0..<28 {
            var r = ""
            for _ in 0..<28 {
                mnist_input [midx] = ctxdata [midx] == 0 ? 0.0 : 1.0
                r += ctxdata [midx] == 0 ? "." : "x"
                midx += 1
            }
            print ("\(r)\n")
        }
        result = mnist_run (mnist)
        print ("Result: \(result)")
        let mnist_guesses = mnist_get_results(mnist, &size)!
        for i in 0..<10 {
            guesses [i] = mnist_guesses [i]
        }
    }
    
    var body: some View {
        GeometryReader { geometry in
            Path { path in
                for drawing in self.drawings {
                    self.add(drawing: drawing, toPath: &path)
                }
                self.add(drawing: self.currentDrawing, toPath: &path)
            }
            .stroke(Color.black, lineWidth: 3)
                .background(Color(white: 0.95))
                .gesture(
                    DragGesture(minimumDistance: 0.1)
                        .onChanged({ (value) in
                            let currentPoint = value.location
                            if currentPoint.y >= 0
                                && currentPoint.y < geometry.size.height {
                                self.currentDrawing.points.append(currentPoint)
                            }
                            self.updateGuess ()
                        })
                        .onEnded({ (value) in
                            self.drawings.append(self.currentDrawing)
                            self.currentDrawing = Drawing()
                            self.updateGuess ()
                        })
            )
        }
        .frame(maxHeight: .infinity)
    }
    
    private func add(drawing: Drawing, toPath path: inout Path) {
        let points = drawing.points
        if points.count > 1 {
            for i in 0..<points.count-1 {
                let current = points[i]
                let next = points[i+1]
                path.move(to: current)
                path.addLine(to: next)
            }
        }
    }
    
}

struct ContentView: View {
    @State private var currentDrawing: Drawing = Drawing()
    @State private var drawings: [Drawing] = [Drawing]()
    @State private var lineWidth: CGFloat = 3.0
    @State var guesses: [Float] = Array.init(repeating: 0, count: 10)
    @State var result: Int = 0
    

    func fmt (_ idx: Int)->String
    {
        let v = guesses [idx]
        return (v >= 0 ? " " : "") + String(format: "%2.2f", v)
    }
    
    var body: some View {
        VStack(alignment: .center) {
            Text("Draw a digit")
                .font(.largeTitle)
            HStack {
                DrawingPad(
                    currentDrawing: $currentDrawing,
                    drawings: $drawings,
                    guesses: $guesses,
                    result: $result)
                    .frame(width: 24*factor, height: 24*factor)
                    .border(Color.black)
                VStack {
                    ForEach (guesses.indices) { idx in
                        HStack {
                        Text ("x")
                        Rectangle ()
                            .offset (x: 10)
                            .border(Color.red)
                            .frame(width:40, height: 20)
                        }
                    }
                }.frame (width: 120)
                VStack (alignment: .leading) {
                    Text ("Result: \(result)")
                    ForEach (guesses.indices) { idx in

                        Text ("\(idx) -> \(self.fmt (idx))")
                        
                    }
                }.frame(width: 120)
            }
            .font(.system(Font.TextStyle.body, design: .monospaced))

            Button(action: {self.drawings = []; self.guesses = Array.init (repeating: 0, count: 10)}) {
                Text ("Clear")
            }
        }
    }
}


struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
