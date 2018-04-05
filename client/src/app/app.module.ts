import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { RouterModule } from "@angular/router";
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { HttpModule } from '@angular/http';

import { AngularFontAwesomeModule } from 'angular-font-awesome';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';

import { MessagesModule } from 'primeng/primeng';
import { MessageModule } from 'primeng/primeng';

import { MatButtonModule, MatProgressBarModule } from '@angular/material';

import { AppComponent } from './app.component';
import { NavbarComponent } from './layout/navbar/navbar.component';
import { HomeComponent } from './pages/home/home.component';
import { UploadComponent } from './pages/upload/upload.component';
import { AboutComponent } from './pages/about/about.component';

import { UploadService } from './services/upload.service';
import { MessageService } from 'primeng/components/common/messageservice';

import { routes } from './app.routes';


@NgModule({
  declarations: [
    AppComponent,
    NavbarComponent,
    HomeComponent,
    UploadComponent,
    AboutComponent
  ],
  imports: [
    BrowserModule,
    FormsModule,
    ReactiveFormsModule,
    HttpModule,
    AngularFontAwesomeModule,
    BrowserAnimationsModule,
    MessagesModule,
    MessageModule,
    MatButtonModule,
    MatProgressBarModule,
    RouterModule.forRoot(routes)
  ],
  providers: [
    UploadService,
    MessageService
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
