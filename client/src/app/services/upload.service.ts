import { Injectable } from '@angular/core';
import { Http, RequestOptions, ResponseContentType } from '@angular/http';
import { Observable } from 'rxjs/Observable';

import 'rxjs/add/operator/map'
import 'rxjs/add/operator/catch';

@Injectable()
export class UploadService {

    private readonly API_URL: string = 'http://localhost:8080/upload';
    private readonly FILE_NAME: string = 'DEEP_SIGHT_FILE';

    constructor(private http: Http) { }

    public upload(file: File, video: boolean) : Observable<any> {

        let options = video ? new RequestOptions({ responseType: ResponseContentType.Blob }) : null ;
        let formData:FormData = new FormData();
        formData.append(this.FILE_NAME, file, file.name);

        return this.http.post(this.API_URL, formData, options);
    }
}
