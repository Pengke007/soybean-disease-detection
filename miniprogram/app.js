App({
  globalData: {
    // 后端服务地址。部署时将此地址改为你的服务器域名（需在微信公众平台配置 request 合法域名）
    apiBase: 'http://localhost:5000',
    // 开发调试时可勾选「不校验合法域名、web-view、TLS 版本以及 HTTPS 证书」
    diseaseLib: {}
  },
  onLaunch() {
    this.loadDiseases();
  },
  loadDiseases() {
    wx.request({
      url: this.globalData.apiBase + '/api/diseases',
      success: (res) => {
        if (res.statusCode === 200 && res.data) this.globalData.diseaseLib = res.data;
      },
      fail: () => {}
    });
  }
});
