import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
import { UploadService } from '../../services/upload.service';
import { Message } from 'primeng/components/common/api';
import { MessageService } from 'primeng/components/common/messageservice';

import 'rxjs/add/observable/throw';


enum Phase {
    Idle = 10,
    Processing = 20,
    Completed = 30,
}


@Component({
    selector: 'ds-upload',
    templateUrl: './upload.component.html',
    styleUrls: ['./upload.component.css']
})
export class UploadComponent implements OnInit {

    @ViewChild('canvas') canvasRef: ElementRef;
    @ViewChild('browseBtn') browseBtnRef: ElementRef;

    // Public members
    public file: File;
    public video: boolean;
    public phase: Phase = Phase.Idle;
    public image_src: string;
    public image_height: number;

    // Private consts
    private readonly FONT_SIZE_FACTOR: number = 0.8;
    private readonly LINE_WIDTH_FACTOR: number = 0.3;
    private readonly VIDEO_EXTENSIONS: Array<string> = ['avi', 'mp4'];

    constructor(private $upload: UploadService,
                private $message: MessageService) { 
    }

    ngOnInit() {
    }


    public browse() : void {
        let browsBtn = this.browseBtnRef.nativeElement;
        browsBtn.click();
    }

    public fileChange(event) {

        let fileList: FileList = event.target.files;

        if(fileList.length > 0) {
            // Change phase to processing and check whether file is a video
            this.phase = Phase.Processing;
            this.file = fileList[0];
            this.video = this.isVideo(this.file);
            
            // Clear canvas and draw image
            this.clearCanvas();

            // Upload file
            this.$upload
                .upload(this.file, this.video)
                .subscribe(
                    response => this.onSuccess(response),
                    error => this.onError(error)
                )
        }
    }


    private clearCanvas() : void {
        
        // Clear canvas and redraw image
        let canvas = this.canvasRef.nativeElement;
        let context = canvas.getContext('2d');
        let image = new Image();
        let url =  URL.createObjectURL(this.file);

        image.onload = () => {
            canvas.width = image.naturalWidth;
            canvas.height = image.naturalHeight;
            this.image_height = image.naturalHeight;
            context.clearRect(0, 0, image.naturalWidth, image.naturalHeight);
            context.drawImage(image, 0, 0);
        }
        
        image.src = url;
    }


    private onSuccess(response: any) : void {

        // Change phase to completed
        this.phase = Phase.Completed;

        // Handle response accordingly to type of input file
        if (this.video) {
            this.handleVideoResponse(response);
        } else {
            this.handleImageResponse(response);
        }
    }
    

    private onError(error: any) : void {

        // Show message and change phase to idle
        this.showMessage('error', 'Error has occured!', error);
        this.phase = Phase.Idle;
    }


    private handleVideoResponse(response: any) : void {

        // Save file
        this.saveVideo(response);
    }

    
    private saveVideo(response) : void {

        var anchor = document.createElement('a');
        var blob = response._body;
        anchor.href = URL.createObjectURL(blob);
        anchor.download = this.file.name;
        anchor.click();  
    }


    private handleImageResponse(response: any) : void {

        // Set refernces
        let canvas = this.canvasRef.nativeElement;
        let context = canvas.getContext('2d');

        // Unpack response with bounding boxes
        let objects: Array<any> = response.json();

        // Draw bounding boxes
        for (var object of objects)
        {
            // Unpack values
            let bbox = object.bbox;
            let x: number = bbox.x;
            let y: number = bbox.y;
            let w: number = bbox.w;
            let h: number = bbox.h;

            // Draw bounding box
            let lineWidth = Math.floor(this.LINE_WIDTH_FACTOR * Math.sqrt(this.image_height));
            let color = object.color;
            context.beginPath();
            context.rect(x, y, w, h);
            context.lineWidth = lineWidth;
            context.strokeStyle = color;
            context.stroke();
            
            // Draw label
            let fontSize =  Math.floor(this.FONT_SIZE_FACTOR * Math.sqrt(this.image_height)); 
            let text: string = object.label + ' (' + object.confidence + ')'
            context.font = 'bold ' + fontSize  + 'px Open Sans';
            context.fillStyle = color;
            context.strokeStyle = 'white';
            context.fillText(text, x, y - fontSize / 2.0);
        } 

        // Save canvas as image
        this.saveCanvas(canvas, this.file.name);
    }


    private saveCanvas(canvas, filename) : void {
        
        // Create a download link and attach the image data URL
        let anchor = document.createElement('a');
        anchor.href = canvas.toDataURL();
        anchor.download = filename;
        anchor.click();     
    }


    private isVideo(file: File) : boolean {  

        for (var extension of this.VIDEO_EXTENSIONS) {
            if (this.file.name.endsWith(extension)) {
                return true;
            }
        }

        return false;
    }

    
    private showMessage(severity: string, summary: string, detail: string) : void {

        this.$message.add({ severity: severity, summary: summary, detail: detail });
        setTimeout(() => this.$message.clear(), 5000);
    }
}
