const app = getApp()

Page({
  data: {
    imagePath: '',
    loading: false,
    result: null
  },

  chooseImage() {
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['camera', 'album'],
      camera: 'back',
      success: (res) => {
        this.setData({ imagePath: res.tempFiles[0].tempFilePath, result: null })
      }
    })
  },

  clearImage() {
    this.setData({ imagePath: '', result: null })
  },

  detect() {
    if (!this.data.imagePath) return
    this.setData({ loading: true })
    wx.showLoading({ title: '识别中...' })

    wx.uploadFile({
      url: app.globalData.apiBase + '/api/predict',
      filePath: this.data.imagePath,
      name: 'image',
      success: (res) => {
        wx.hideLoading()
        try {
          const data = JSON.parse(res.data)
          if (data.error) throw new Error(data.error)
          this.setData({ result: data })
        } catch (e) {
          wx.showToast({ title: '解析失败', icon: 'none' })
        }
      },
      fail: () => {
        wx.hideLoading()
        wx.showModal({
          title: '连接失败',
          content: '无法连接识别服务。请确认：\n1. 后端服务已启动\n2. 开发者工具已勾选「不校验合法域名」\n3. apiBase 地址正确',
          showCancel: false
        })
      },
      complete: () => this.setData({ loading: false })
    })
  },

  goDetail(e) {
    wx.navigateTo({ url: '/pages/detail/detail?cls=' + e.currentTarget.dataset.cls })
  }
})
