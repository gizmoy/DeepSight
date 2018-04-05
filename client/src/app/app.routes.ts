import { Routes } from "@angular/router";

import { HomeComponent } from './pages/home/home.component';
import { UploadComponent } from './pages/upload/upload.component';
import { AboutComponent } from './pages/about/about.component';



export const routes: Routes = [
    { path: '', redirectTo: 'home', pathMatch: 'full' },
    { path: 'home', component: HomeComponent },
    { path: 'upload', component: UploadComponent },
    { path: 'about', component: AboutComponent }
];