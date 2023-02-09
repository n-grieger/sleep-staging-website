/*! For license information please see 858.js.LICENSE.txt */
!function(t,e){if("object"==typeof exports&&"object"==typeof module)module.exports=e();else if("function"==typeof define&&define.amd)define([],e);else{var r=e();for(var n in r)("object"==typeof exports?exports:t)[n]=r[n]}}(self,(function(){return function(){"use strict";var t,e,r={858:function(t,e,r){function n(t,e){var r=t.length,n=new Float32Array(r+e.length);return n.set(t),n.set(e,r),n}function o(t){return o="function"==typeof Symbol&&"symbol"==typeof Symbol.iterator?function(t){return typeof t}:function(t){return t&&"function"==typeof Symbol&&t.constructor===Symbol&&t!==Symbol.prototype?"symbol":typeof t},o(t)}function i(){i=function(){return t};var t={},e=Object.prototype,r=e.hasOwnProperty,n=Object.defineProperty||function(t,e,r){t[e]=r.value},a="function"==typeof Symbol?Symbol:{},c=a.iterator||"@@iterator",u=a.asyncIterator||"@@asyncIterator",s=a.toStringTag||"@@toStringTag";function l(t,e,r){return Object.defineProperty(t,e,{value:r,enumerable:!0,configurable:!0,writable:!0}),t[e]}try{l({},"")}catch(t){l=function(t,e,r){return t[e]=r}}function f(t,e,r,o){var i=e&&e.prototype instanceof y?e:y,a=Object.create(i.prototype),c=new E(o||[]);return n(a,"_invoke",{value:_(t,r,c)}),a}function p(t,e,r){try{return{type:"normal",arg:t.call(e,r)}}catch(t){return{type:"throw",arg:t}}}t.wrap=f;var h={};function y(){}function d(){}function g(){}var v={};l(v,c,(function(){return this}));var m=Object.getPrototypeOf,b=m&&m(m(P([])));b&&b!==e&&r.call(b,c)&&(v=b);var w=g.prototype=y.prototype=Object.create(v);function x(t){["next","throw","return"].forEach((function(e){l(t,e,(function(t){return this._invoke(e,t)}))}))}function O(t,e){function i(n,a,c,u){var s=p(t[n],t,a);if("throw"!==s.type){var l=s.arg,f=l.value;return f&&"object"==o(f)&&r.call(f,"__await")?e.resolve(f.__await).then((function(t){i("next",t,c,u)}),(function(t){i("throw",t,c,u)})):e.resolve(f).then((function(t){l.value=t,c(l)}),(function(t){return i("throw",t,c,u)}))}u(s.arg)}var a;n(this,"_invoke",{value:function(t,r){function n(){return new e((function(e,n){i(t,r,e,n)}))}return a=a?a.then(n,n):n()}})}function _(t,e,r){var n="suspendedStart";return function(o,i){if("executing"===n)throw new Error("Generator is already running");if("completed"===n){if("throw"===o)throw i;return k()}for(r.method=o,r.arg=i;;){var a=r.delegate;if(a){var c=S(a,r);if(c){if(c===h)continue;return c}}if("next"===r.method)r.sent=r._sent=r.arg;else if("throw"===r.method){if("suspendedStart"===n)throw n="completed",r.arg;r.dispatchException(r.arg)}else"return"===r.method&&r.abrupt("return",r.arg);n="executing";var u=p(t,e,r);if("normal"===u.type){if(n=r.done?"completed":"suspendedYield",u.arg===h)continue;return{value:u.arg,done:r.done}}"throw"===u.type&&(n="completed",r.method="throw",r.arg=u.arg)}}}function S(t,e){var r=e.method,n=t.iterator[r];if(void 0===n)return e.delegate=null,"throw"===r&&t.iterator.return&&(e.method="return",e.arg=void 0,S(t,e),"throw"===e.method)||"return"!==r&&(e.method="throw",e.arg=new TypeError("The iterator does not provide a '"+r+"' method")),h;var o=p(n,t.iterator,e.arg);if("throw"===o.type)return e.method="throw",e.arg=o.arg,e.delegate=null,h;var i=o.arg;return i?i.done?(e[t.resultName]=i.value,e.next=t.nextLoc,"return"!==e.method&&(e.method="next",e.arg=void 0),e.delegate=null,h):i:(e.method="throw",e.arg=new TypeError("iterator result is not an object"),e.delegate=null,h)}function L(t){var e={tryLoc:t[0]};1 in t&&(e.catchLoc=t[1]),2 in t&&(e.finallyLoc=t[2],e.afterLoc=t[3]),this.tryEntries.push(e)}function j(t){var e=t.completion||{};e.type="normal",delete e.arg,t.completion=e}function E(t){this.tryEntries=[{tryLoc:"root"}],t.forEach(L,this),this.reset(!0)}function P(t){if(t){var e=t[c];if(e)return e.call(t);if("function"==typeof t.next)return t;if(!isNaN(t.length)){var n=-1,o=function e(){for(;++n<t.length;)if(r.call(t,n))return e.value=t[n],e.done=!1,e;return e.value=void 0,e.done=!0,e};return o.next=o}}return{next:k}}function k(){return{value:void 0,done:!0}}return d.prototype=g,n(w,"constructor",{value:g,configurable:!0}),n(g,"constructor",{value:d,configurable:!0}),d.displayName=l(g,s,"GeneratorFunction"),t.isGeneratorFunction=function(t){var e="function"==typeof t&&t.constructor;return!!e&&(e===d||"GeneratorFunction"===(e.displayName||e.name))},t.mark=function(t){return Object.setPrototypeOf?Object.setPrototypeOf(t,g):(t.__proto__=g,l(t,s,"GeneratorFunction")),t.prototype=Object.create(w),t},t.awrap=function(t){return{__await:t}},x(O.prototype),l(O.prototype,u,(function(){return this})),t.AsyncIterator=O,t.async=function(e,r,n,o,i){void 0===i&&(i=Promise);var a=new O(f(e,r,n,o),i);return t.isGeneratorFunction(r)?a:a.next().then((function(t){return t.done?t.value:a.next()}))},x(w),l(w,s,"Generator"),l(w,c,(function(){return this})),l(w,"toString",(function(){return"[object Generator]"})),t.keys=function(t){var e=Object(t),r=[];for(var n in e)r.push(n);return r.reverse(),function t(){for(;r.length;){var n=r.pop();if(n in e)return t.value=n,t.done=!1,t}return t.done=!0,t}},t.values=P,E.prototype={constructor:E,reset:function(t){if(this.prev=0,this.next=0,this.sent=this._sent=void 0,this.done=!1,this.delegate=null,this.method="next",this.arg=void 0,this.tryEntries.forEach(j),!t)for(var e in this)"t"===e.charAt(0)&&r.call(this,e)&&!isNaN(+e.slice(1))&&(this[e]=void 0)},stop:function(){this.done=!0;var t=this.tryEntries[0].completion;if("throw"===t.type)throw t.arg;return this.rval},dispatchException:function(t){if(this.done)throw t;var e=this;function n(r,n){return a.type="throw",a.arg=t,e.next=r,n&&(e.method="next",e.arg=void 0),!!n}for(var o=this.tryEntries.length-1;o>=0;--o){var i=this.tryEntries[o],a=i.completion;if("root"===i.tryLoc)return n("end");if(i.tryLoc<=this.prev){var c=r.call(i,"catchLoc"),u=r.call(i,"finallyLoc");if(c&&u){if(this.prev<i.catchLoc)return n(i.catchLoc,!0);if(this.prev<i.finallyLoc)return n(i.finallyLoc)}else if(c){if(this.prev<i.catchLoc)return n(i.catchLoc,!0)}else{if(!u)throw new Error("try statement without catch or finally");if(this.prev<i.finallyLoc)return n(i.finallyLoc)}}}},abrupt:function(t,e){for(var n=this.tryEntries.length-1;n>=0;--n){var o=this.tryEntries[n];if(o.tryLoc<=this.prev&&r.call(o,"finallyLoc")&&this.prev<o.finallyLoc){var i=o;break}}i&&("break"===t||"continue"===t)&&i.tryLoc<=e&&e<=i.finallyLoc&&(i=null);var a=i?i.completion:{};return a.type=t,a.arg=e,i?(this.method="next",this.next=i.finallyLoc,h):this.complete(a)},complete:function(t,e){if("throw"===t.type)throw t.arg;return"break"===t.type||"continue"===t.type?this.next=t.arg:"return"===t.type?(this.rval=this.arg=t.arg,this.method="return",this.next="end"):"normal"===t.type&&e&&(this.next=e),h},finish:function(t){for(var e=this.tryEntries.length-1;e>=0;--e){var r=this.tryEntries[e];if(r.finallyLoc===t)return this.complete(r.completion,r.afterLoc),j(r),h}},catch:function(t){for(var e=this.tryEntries.length-1;e>=0;--e){var r=this.tryEntries[e];if(r.tryLoc===t){var n=r.completion;if("throw"===n.type){var o=n.arg;j(r)}return o}}throw new Error("illegal catch attempt")},delegateYield:function(t,e,r){return this.delegate={iterator:P(t),resultName:e,nextLoc:r},"next"===this.method&&(this.arg=void 0),h}},t}function a(t){return function(t){if(Array.isArray(t))return c(t)}(t)||function(t){if("undefined"!=typeof Symbol&&null!=t[Symbol.iterator]||null!=t["@@iterator"])return Array.from(t)}(t)||function(t,e){if(!t)return;if("string"==typeof t)return c(t,e);var r=Object.prototype.toString.call(t).slice(8,-1);"Object"===r&&t.constructor&&(r=t.constructor.name);if("Map"===r||"Set"===r)return Array.from(t);if("Arguments"===r||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r))return c(t,e)}(t)||function(){throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}()}function c(t,e){(null==e||e>t.length)&&(e=t.length);for(var r=0,n=new Array(e);r<e;r++)n[r]=t[r];return n}function u(t,e,r,n,o,i,a){try{var c=t[i](a),u=c.value}catch(t){return void r(t)}c.done?e(u):Promise.resolve(u).then(n,o)}function s(t){return function(){var e=this,r=arguments;return new Promise((function(n,o){var i=t.apply(e,r);function a(t){u(i,n,o,a,c,"next",t)}function c(t){u(i,n,o,a,c,"throw",t)}a(void 0)}))}}var l=r(550),f=r(258).loadPyodide,p=r(736);function h(t){postMessage([0,t])}function y(){return(y=s(i().mark((function t(e){var r,o,c,u,s,y,d,g,v,m,b,w,x,O,_,S,L,j,E,P,k,T,A,N,F,M,G,I,C,J,$,R,U,Y,q,B;return i().wrap((function(t){for(;;)switch(t.prev=t.next){case 0:for(r=e.data_buffer,o=e.channels,c=e.model,u=e.sampling_rate,s=e.preprocessing_script,y=e.postprocessing_script,d=10,g=4,(v=new l.EdfDecoder).setInput(r),v.decode(),m=v.getOutput(),b=[],w=0;w<m.getNumberOfSignals();w++)b.push(m.getSignalLabel(w));for(h(5),x=m.getNumberOfRecords(),O=[],_=[],S=0;S<o.length;S++)L=m.getPhysicalSignalConcatRecords(b.indexOf(o[S]),0,x),_.push(new Float32Array(L)),O.push(m.getSignalSamplingFrequency(b.indexOf(o[S])));return h(8),t.next=22,f({indexURL:"./pyodide"});case 22:return j=t.sent,t.next=25,fetch("./"+s);case 25:return t.next=27,t.sent.text();case 27:return E=t.sent,h(11),t.prev=29,t.next=32,j.loadPackage("numpy");case 32:return t.next=34,j.loadPackage("scikit-learn");case 34:return t.next=36,j.loadPackage("scipy");case 36:return j.globals.set("sampling_rates",O),j.globals.set("preprocessing_sampling_rate",u),j.globals.set("channel_data",_),t.t0=j,t.next=42,fetch("./preprocessing_record.py");case 42:return t.next=44,t.sent.text();case 44:t.t1=t.sent,t.t0.runPython.call(t.t0,t.t1),t.next=51;break;case 48:t.prev=48,t.t2=t.catch(29),console.log(t.t2);case 51:return h(15),p.env.wasm.wasmPaths="./",p.env.wasm.numThreads=4,P={executionProviders:["wasm"]},t.next=57,p.InferenceSession.create("./models/".concat(c,".with_runtime_opt.ort"),P);case 57:k=t.sent,h(20),T=j.globals.get("samples").toJs(),A=[],N=0,F=0;case 63:if(!(F<g)){t.next=90;break}M=Math.floor(T.length/g),F===g-1&&(M=T.length-N);try{j.globals.set("channel_data",T),j.globals.set("start_index",N),j.globals.set("n_samples",M),j.runPython(E)}catch(t){console.log(t)}G=j.globals.get("samples").toJs(),I=j.globals.get("input_shape").toJs(),C=0;case 70:if(!(C<M)){t.next=87;break}for(J=M-C<d?M-C:d,$=new Float32Array(G[C]),R=1;R<J;R++)$=n($,G[C+R]);return U=new p.Tensor("float32",$,[J].concat(I)),t.next=77,k.run({input:U});case 77:for(Y=t.sent,q=Y.output.data,B=0;B<q.length;B+=5)A.push(q.slice(B,B+5).indexOf(Math.max.apply(Math,a(q.slice(B,B+5)))));return h(20+Math.round(N/T.length*75)),N+=J,t.next=84,new Promise((function(t){return setTimeout(t,0)}));case 84:C+=d,t.next=70;break;case 87:F+=1,t.next=63;break;case 90:return t.prev=90,j.globals.set("labels_in",A),t.t3=j,t.next=95,fetch("./"+y);case 95:return t.next=97,t.sent.text();case 97:t.t4=t.sent,t.t3.runPython.call(t.t3,t.t4),A=j.globals.get("labels_out").toJs(),t.next=106;break;case 102:return t.prev=102,t.t5=t.catch(90),console.log(t.t5),t.abrupt("return",[0]);case 106:return h(0),t.abrupt("return",A);case 108:case"end":return t.stop()}}),t,null,[[29,48],[90,102]])})))).apply(this,arguments)}onmessage=function(t){(function(t){return y.apply(this,arguments)})(t.data).then((function(t){postMessage([1,t])}))}}},n={};function o(t){var e=n[t];if(void 0!==e)return e.exports;var i=n[t]={exports:{}};return r[t].call(i.exports,i,i.exports,o),i.exports}return o.m=r,o.x=function(){var t=o.O(void 0,[84],(function(){return o(858)}));return t=o.O(t)},t=[],o.O=function(e,r,n,i){if(!r){var a=1/0;for(l=0;l<t.length;l++){r=t[l][0],n=t[l][1],i=t[l][2];for(var c=!0,u=0;u<r.length;u++)(!1&i||a>=i)&&Object.keys(o.O).every((function(t){return o.O[t](r[u])}))?r.splice(u--,1):(c=!1,i<a&&(a=i));if(c){t.splice(l--,1);var s=n();void 0!==s&&(e=s)}}return e}i=i||0;for(var l=t.length;l>0&&t[l-1][2]>i;l--)t[l]=t[l-1];t[l]=[r,n,i]},o.d=function(t,e){for(var r in e)o.o(e,r)&&!o.o(t,r)&&Object.defineProperty(t,r,{enumerable:!0,get:e[r]})},o.f={},o.e=function(t){return Promise.all(Object.keys(o.f).reduce((function(e,r){return o.f[r](t,e),e}),[]))},o.u=function(t){return t+".js"},o.g=function(){if("object"==typeof globalThis)return globalThis;try{return this||new Function("return this")()}catch(t){if("object"==typeof window)return window}}(),o.o=function(t,e){return Object.prototype.hasOwnProperty.call(t,e)},o.r=function(t){"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(t,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(t,"__esModule",{value:!0})},function(){var t;o.g.importScripts&&(t=o.g.location+"");var e=o.g.document;if(!t&&e&&(e.currentScript&&(t=e.currentScript.src),!t)){var r=e.getElementsByTagName("script");r.length&&(t=r[r.length-1].src)}if(!t)throw new Error("Automatic publicPath is not supported in this browser");t=t.replace(/#.*$/,"").replace(/\?.*$/,"").replace(/\/[^\/]+$/,"/"),o.p=t}(),function(){var t={858:1};o.f.i=function(e,r){t[e]||importScripts(o.p+o.u(e))};var e=self.webpackChunkv4=self.webpackChunkv4||[],r=e.push.bind(e);e.push=function(e){var n=e[0],i=e[1],a=e[2];for(var c in i)o.o(i,c)&&(o.m[c]=i[c]);for(a&&a(o);n.length;)t[n.pop()]=1;r(e)}}(),e=o.x,o.x=function(){return o.e(84).then(e)},o.x()}()}));